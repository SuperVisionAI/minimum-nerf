import torch
import numpy as np
from matplotlib.image import imread, imsave

class NerfNetwork(torch.nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_dir=4, hidden_dim=128):   
        super().__init__()
        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_dir = embedding_dim_dir
        self.block1 = torch.nn.Sequential(torch.nn.Linear(embedding_dim_pos * 6 + 3, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU())
        self.block2 = torch.nn.Sequential(torch.nn.Linear(embedding_dim_pos * 6 + hidden_dim + 3, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Linear(hidden_dim, hidden_dim + 1))
        self.block3 = torch.nn.Sequential(torch.nn.Linear(embedding_dim_dir * 6 + hidden_dim + 3, hidden_dim // 2), torch.nn.ReLU())
        self.block4 = torch.nn.Sequential(torch.nn.Linear(hidden_dim // 2, 3), torch.nn.Sigmoid())

    def forward(self, o, d):
        def positional_encoding(x, l):
	        out = [x]
	        for j in range(l):
	            out.append(torch.sin(2 ** j * x))
	            out.append(torch.cos(2 ** j * x))
	        out = torch.cat(out, dim=1)
	        return out

        emb_x = positional_encoding(o, self.embedding_dim_pos) # emb_x: [batch_size, embedding_dim_pos * 6]
        emb_d = positional_encoding(d, self.embedding_dim_dir) # emb_d: [batch_size, embedding_dim_dir * 6]
        h = self.block1(emb_x) # h: [batch_size, hidden_dim]
        t = self.block2(torch.cat((h, emb_x), dim=1)) # tmp: [batch_size, hidden_dim + 1]
        h, sigma = t[:, :-1], torch.nn.functional.relu(t[:, -1]) # h: [batch_size, hidden_dim], sigma: [batch_size]
        h = self.block3(torch.cat((h, emb_d), dim=1)) # h: [batch_size, hidden_dim // 2]
        c = self.block4(h) # c: [batch_size, 3]
        return c, sigma

def render_rays(nerf_network, ray_origins, ray_directions, hn, hf, nb_bins):
    def compute_accumulated_transmittance(alphas):
        accumulated_transmittance = torch.cumprod(alphas, 1)
        return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device), accumulated_transmittance[:, :-1]), dim=-1)
    t = torch.linspace(hn, hf, nb_bins, device=ray_origins.device).expand(ray_origins.shape[0], nb_bins)
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=ray_origins.device)  #perturb sampling along each ray
    t = lower + (upper - lower) * u  #[batch_size, nb_bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=ray_origins.device).expand(ray_origins.shape[0], 1)), -1)
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1)  #compute the 3D points along each ray  #[batch_size, nb_bins, 3]  
    ray_directions = ray_directions.expand(nb_bins, ray_directions.shape[0], 3).transpose(0, 1)  #expand the ray_directions tensor to match the shape of x
    colors, sigma = nerf_network(x.reshape(-1, 3), ray_directions.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma = sigma.reshape(x.shape[:-1])
    alpha = 1 - torch.exp(-sigma * delta)  #[batch_size, nb_bins]
    weights = compute_accumulated_transmittance(1 - alpha).unsqueeze(2) * alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)  #pixel values
    weight_sum = weights.sum(-1).sum(-1)  #regularization for white background
    return c + 1 - weight_sum.unsqueeze(-1)

def train(batch_size, H, W, steps, dataset, nerf_network, hn, hf, nb_bins=192, device=None):
    optimizer = torch.optim.Adam(nerf_network.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_size = len(dataset)
    training_loss = []
    for s in range(steps):
        for b,batch in enumerate(data_loader):
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            ground_truth_px_values = batch[:, 6:].to(device)            
            regenerated_px_values = render_rays(nerf_network, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins) 
            loss = ((ground_truth_px_values - regenerated_px_values) ** 2).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
            if b%(100)==0: 
                print('train', 'step=%04d/%04d  batch=%04d/%06d'%(s,steps, b, data_size//batch_size), '', 'loss=%.4f'%(loss.item()))
        scheduler.step()
        for img_index in range(200):
            valid(hn, hf, testing_dataset, img_index=img_index, nb_bins=nb_bins, H=H, W=W)
    return training_loss

@torch.no_grad()
def valid(hn, hf, dataset, chunk_size=10, img_index=0, nb_bins=192, H=400, W=400):
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]
    data = []   # list of regenerated pixel values
    for i in range(int(np.ceil(H / chunk_size))):   # iterate over chunks
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)        
        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    image = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)
    imsave('./outs/valid_%03d.png'%(img_index), image)

def nerf(device=('cuda' if torch.cuda.is_available() else 'cpu')):
    training_dataset = torch.from_numpy(np.load('./data/100L/training_data.pkl', allow_pickle=True))  #[16000000, 9] = [16000000=batch/100*H/400*W/400, 9=origin/3+direction/3+rgb/3]
    testing_dataset = torch.from_numpy(np.load('./data/100L/testing_data.pkl', allow_pickle=True))  #https://drive.google.com/drive/folders/18bwm-RiHETRCS5yD9G00seFIcrJHIvD-?usp=sharing
    print('training_dataset', training_dataset.shape)
    print('testing_dataset', testing_dataset.shape)    
    nerf_network = NerfNetwork(hidden_dim=256).to(device)
    train(batch_size=1024, H=400, W=400, steps=16, dataset=training_dataset, nerf_network=nerf_network, hn=2, hf=6, nb_bins=192, device=device)

if __name__ == '__main__':
    nerf()