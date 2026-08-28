import os
import pygame

def create_texture_atlas(asset_path):
    categories = ['block', 'item', 'destroy_stage', 'particle', "pickaxe"]
    textures = {category: {} for category in categories}
    images = []
    positions = {}
    atlas_width = 512

    # Load images and track dimensions
    x_offset, y_offset, row_height = 0, 0, 0
    for category in categories:
        folder_path = os.path.join(asset_path, category)
        if not os.path.exists(folder_path):
            print("Folder not found: ", folder_path)
            continue
        
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".png"):
                img_path = os.path.join(folder_path, filename)
                image = pygame.image.load(img_path).convert_alpha()
                img_width, img_height = image.get_size()
                
                # Wrap to new row if necessary
                if x_offset + img_width > atlas_width:  # Max texture width
                    x_offset = 0
                    y_offset += row_height
                    row_height = 0
                
                # Store texture coordinates
                texture_name = filename.rsplit(".", 1)[0]
                textures[category][texture_name] = (x_offset, y_offset, img_width, img_height)
                
                images.append((image, (x_offset, y_offset)))
                
                x_offset += img_width
                row_height = max(row_height, img_height)
    
    # Determine atlas size
    atlas_height = y_offset + row_height
    atlas_surface = pygame.Surface((atlas_width, atlas_height), pygame.SRCALPHA)
    
    # Blit images onto atlas
    for image, pos in images:
        atlas_surface.blit(image, pos)
    
    return atlas_surface, textures
