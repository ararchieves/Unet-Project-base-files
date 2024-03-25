import torch 
import matplotlib.pyplot as plt

from .general import split_classes

def visualize_results(loader, model, idx=0, thres=[0.5, 0.6]):
    assert len(thres) == 2, f"Only 2 threshold values are supported!"
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    
    images, houses, blocks = next(iter(loader))
    image, house, block = images[idx], houses[idx], blocks[idx]
    
    with torch.no_grad():
        out = model(image.unsqueeze(0).to(device))
        house_pred, block_pred = split_classes(out)
        house_pred, block_pred = house_pred.squeeze(0).cpu(), block_pred.squeeze(0).cpu()
        
    plt.title("Image")
    plt.imshow(image.permute(1,2,0))
    
    ## Houses 
    fig, ax = plt.subplots(1, 4, figsize=(15, 10))
    ax[0].set_title("True House")
    ax[0].imshow(house.permute(1,2,0))
    
    ax[1].set_title("Pred House")
    ax[1].imshow(house_pred.permute(1,2,0))
    
    thres_pred = torch.where(house_pred > thres[0], 1, 0)
    ax[2].set_title(f"Pred House@{thres[0]}")
    ax[2].imshow(thres_pred.permute(1,2,0))
    
    thres_pred = torch.where(house_pred > thres[1], 1, 0)
    ax[3].set_title(f"Pred House@{thres[1]}")
    ax[3].imshow(thres_pred.permute(1,2,0))
    
    plt.show()
    ## Blocks
    fig, ax = plt.subplots(1, 4, figsize=(15, 10))
    ax[0].set_title("True Blocks")
    ax[0].imshow(block.permute(1,2,0))
    
    ax[1].set_title("Pred Blocks")
    ax[1].imshow(block_pred.permute(1,2,0))
    
    thres_pred = torch.where(block_pred > thres[0], 1, 0)
    ax[2].set_title(f"Pred Blocks@{thres[0]}")
    ax[2].imshow(thres_pred.permute(1,2,0))
    
    thres_pred = torch.where(block_pred > thres[1], 1, 0)
    ax[3].set_title(f"Pred Blocks@{thres[1]}")
    ax[3].imshow(thres_pred.permute(1,2,0))
    
    plt.show()