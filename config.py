param_dict_shelf = {
    'scene_name': 'Shelf',
    'n_views': 30,
    'rgb_weight_exp': 0.3, 
    'fnf_weight_exp': -1,  
    'beta_weight': -1,    
    'beta_gamma': 2,      
    'depth_weight': -2,   
    'depth_gamma': 2,   
    'iterations': 5_000,
    'test_iterations': [5_000],
}

param_dict_poster = {
    'scene_name': 'Poster',
    'n_views': 30,
    'rgb_weight_exp': 0.3, 
    'fnf_weight_exp': -3,  
    'beta_weight': 1,     
    'beta_gamma': 3,      
    'depth_weight': -2,   
    'depth_gamma': 2,   
    'iterations': 5_000,
    'test_iterations': [5_000],
}

param_dict_lens_stage = {
    'scene_name': 'Lens_Stage',
    'n_views': 30,
    'rgb_weight_exp': 0.3, 
    'fnf_weight_exp': -3, 
    'beta_weight': -1,     
    'beta_gamma': 2,     
    'depth_weight': -2,  
    'depth_gamma': 2,   
    'iterations': 5_000,
    'test_iterations': [5_000],
}

param_dict_ps5 = {
    'scene_name': 'PS5',   
    'n_views': 30,
    'rgb_weight_exp': 0, 
    'fnf_weight_exp': -3,  
    'beta_weight': -2.0,   
    'beta_gamma': 0,     
    'depth_weight': -2, 
    'depth_gamma': 2,  
    'iterations': 50_000,
    'test_iterations': [50_000],
}

param_dict_office = {
    'scene_name': 'Office',
    'n_views': 30,
    'rgb_weight_exp': 0.3, 
    'fnf_weight_exp': -3,  # original was -2
    'beta_weight': -2,    
    'beta_gamma': 2,      
    'depth_weight': -2,   
    'depth_gamma': 2,  
    'iterations': 5_000,
    'test_iterations':[5_000],
}

param_dict_outdoor= {
    'scene_name': 'Outdoor',
    'n_views': 30,
    'rgb_weight_exp': 0.3, 
    'fnf_weight_exp': -3, 
    'beta_weight': -1,    
    'beta_gamma': 1,    
    'depth_weight': -4, 
    'depth_gamma': 2,  
    'iterations': 50_000,
    'test_iterations': [50_000],
}

param_dict_default= {
    'scene_name': 'Default_Name',
    'n_views': 30,
    'rgb_weight_exp': 0.3, 
    'fnf_weight_exp': -3, 
    'beta_weight': -1,    
    'beta_gamma': 1,    
    'depth_weight': -4, 
    'depth_gamma': 2,  
    'iterations': 50_000,
    'test_iterations': [50_000],
}

