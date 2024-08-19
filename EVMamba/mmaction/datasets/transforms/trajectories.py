import torch
import torch.nn as nn

class Tracks(nn.Module):
    def __init__(self): 
        super().__init__()
        
    def build__list(self, coor, features):
        coor = torch.tensor(coor, dtype=torch.float32)
        features = torch.tensor(features, dtype=torch.float32)
    
        sorted_indices = torch.argsort(coor[:, 0])  
        coor = coor[sorted_indices]
        features = features[sorted_indices]

        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        trajectories = []
        features_list = []
        
        # visited = set()
        visited_idx = [] 
        for start_idx in range(len(coor)):
            # if start_idx in visited:
            if start_idx in visited_idx:
                continue
            
            current_trajectory = [coor[start_idx].numpy().tolist()]
            features_trajectory = [features[start_idx].numpy().tolist()]
            
            # visited.add(start_idx)
            visited_idx.append(start_idx)
            current_idx = start_idx
            
            while True:
                current_time = coor[current_idx, 0]
                next_time_indices = (coor[:, 0] > current_time).nonzero(as_tuple=True)[0]
                
                if len(next_time_indices) == 0:
                    break

                next_time = coor[next_time_indices[0], 0].item()
                
                next_indices = (coor[:, 0] == next_time).nonzero(as_tuple=True)[0]
                
                next_features = features[next_indices]
                current_feature = features[current_idx].unsqueeze(0)
                
                similarities = cos(current_feature, next_features)
                best_match_idx = next_indices[torch.argmax(similarities)].item()
                
                # if best_match_idx in visited:
                if best_match_idx in visited_idx:
                    break
                # visited.add(best_match_idx)
                visited_idx.append(best_match_idx) 
                
                current_idx = best_match_idx
                current_trajectory.append(coor[current_idx].numpy().tolist())
                features_trajectory.append(features[current_idx].numpy().tolist())
                
            trajectories.append(current_trajectory)
            features_list.append(features_trajectory)
            
        return trajectories, features_list
    
    def forward(self, coor, features, min_length):
        trajectories_list, features_list = self.build__list(coor, features)
        event_voxels = []
        for i in range(len(trajectories_list)):
            trajectories_list[i] = torch.tensor(trajectories_list[i], dtype=torch.float32)
            features_list[i] = torch.tensor(features_list[i], dtype=torch.float32)
            event_voxels.append(torch.cat((trajectories_list[i],features_list[i]),dim=1))
        
        new_list = []
        for event_voxel in event_voxels:
            if len(event_voxel) >= min_length:  # the minimum length of grids that can be retained
                new_list.append(event_voxel)
        if len(new_list) != 0:
            event_features = torch.cat((new_list),dim=0)
        else: 
            event_features = torch.zeros(4096, 131) 
                
        return event_features
        