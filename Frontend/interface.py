# Library imports
import gradio as gr
import numpy as np
import imageio
import os
import json
import gc
from PIL import Image



# Project files imports
from Backend.plot_grids_utils import plot_voxels_numpy, plot_voxels_pyvista
from VAE_Model.evaluate.eval import Eval_VAE
from Backend.app_model import trained_model
from Backend.prepare_model import initialize_model, warm_up_model, preload_objects
from Backend.plot_grids_utils import reconstruct_object, show_morphing_gif
class GradioApp:
    def __init__(self, model):
        self.model = model
        self.category_prews, self.all_previews_list, self.all_preview_images = self.load_previews()
        with gr.Blocks() as self.demo:
            self.build_ui()

    def launch(self):
        #warm_up_model(self.model)
        self.demo.launch()
    def load_previews(self):
        category_prews = {}
        all_previews_list = []
        all_preview_images = []
        print("Loading previews from cache...")
        CACHE_DIR = r"Frontend/previews_cache"

        METADATA_FILE = os.path.join(CACHE_DIR, "previews_metadata.json")
        print(f"Attempting to open metadata file: {METADATA_FILE}")

        # Load the metadata from the JSON file
        with open(METADATA_FILE, 'r') as f:
            cached_metadata = json.load(f)

        for category, items in cached_metadata.items():
            previews = []
            for item in items:
                image_data = imageio.imread(item["image_cache_path"])
                preview_item = {
                    "name": item["name"],
                    "path": item["path"],
                    "image": image_data ,
                    "image_cache_path": item["image_cache_path"]
                }
                previews.append(preview_item)
                all_previews_list.append(preview_item)
                
            category_prews[category] = previews
            print("Previews loaded successfully!")
        return category_prews, all_previews_list, all_preview_images

    def create_reconstruct_callback(self, previews_list):
        """function to create a unique and correct callback for each category tab."""
        def reconstruct_callback(threshold: float, evt: gr.SelectData):
            selected_index = evt.index
            
            # Get the corresponding object from our list
            found_item = previews_list[selected_index]

            # Get the .npy path for reconstruction
            file_path = found_item["path"]
            file_obj = type('File', (), {'name': file_path})()
            return reconstruct_object(self.model, file_obj, threshold)
            
        return reconstruct_callback


    def interpolation_callback(self,item_a, item_b, steps, threshold):
        """Callback that receives the full item data from gr.State."""
        gc.collect()

        if item_a is None or item_b is None:
            raise gr.Error("Please select one object from each gallery (A and B).")
        
        path_a = item_a["path"]
        path_b = item_b["path"]

        file_obj_a = type('File', (), {'name': path_a})()
        file_obj_b = type('File', (), {'name': path_b})()
        return show_morphing_gif(self.model, file_obj_a, file_obj_b, steps, threshold)



    def build_ui(self):    
        gr.Markdown("# 🤖 3D Variational Autoencoder (VAE) Playground")
        with gr.Tabs():
            # Tab 1: Latent Space Interpolation
            with gr.TabItem("✨ Latent Space Interpolation"):
                gr.Markdown("<span style='font-family: Arial; font-size: 24px;'>"
                    "Select an Object A and an Object B to Morph. Use the dropdown above each gallery to filter its contents.")
                gr.Markdown("  <span style='font-family: Arial; font-size: 16px;'>"
                "Click on an image to select it. After selecting one from each gallery, " \
                "click the button below to generate the interpolation animation. " \
                "The threshold slider controls the minimum voxel probability for a voxel to be shown in the plots. " \
                "The number of morphing steps controls how many intermediate frames are generated between" \
                " the two objects.")

                # State variables to hold the full data of the selected objects
                selected_item_A = gr.State(value=None)
                selected_item_B = gr.State(value=None)
                
                category_choices = ["All"] + sorted(list(self.category_prews.keys()))
                all_preview_images = [p['image'] for p in self.all_previews_list]

                with gr.Row():
                    with gr.Column():
                        category_filter_A = gr.Dropdown(choices=category_choices, value="All", label="Filter Gallery A by Category")
                        interpolation_gallery_A = gr.Gallery(value=all_preview_images, 
                                                            label="Select Object A", 
                                                            columns=4, height=360, 
                                                            allow_preview=False)
                    
                    with gr.Column():
                        category_filter_B = gr.Dropdown(choices=category_choices, value="All", label="Filter Gallery B by Category")
                        interpolation_gallery_B = gr.Gallery(value=all_preview_images, 
                                                            label="Select Object B", 
                                                            columns=4, height=360, 
                                                            allow_preview=False)
                
                def update_gallery(category):
                    """Updates a gallery's content based on the selected category."""
                    if category == "All":
                        return gr.update(value=all_preview_images)
                    else:
                        images_to_show = [p['image'] for p in self.category_prews[category]]
                        return gr.update(value=images_to_show)

                def store_selection(category, evt: gr.SelectData):
                    """Finds the full data for the selected item and returns it to be stored in State."""
                    if category == "All":
                        return self.all_previews_list[evt.index]
                    else:
                        return self.category_prews[category][evt.index]
                
                # When a filter dropdown changes, update its corresponding gallery
                category_filter_A.change(fn=update_gallery, inputs=[category_filter_A], outputs=[interpolation_gallery_A])
                category_filter_B.change(fn=update_gallery, inputs=[category_filter_B], outputs=[interpolation_gallery_B])
                
                # When an image is selected, store its full data in the appropriate state variable
                interpolation_gallery_A.select(fn=store_selection, inputs=[category_filter_A], outputs=[selected_item_A])
                interpolation_gallery_B.select(fn=store_selection, inputs=[category_filter_B], outputs=[selected_item_B])

                with gr.Row():
                    threshold_slider_interp = gr.Slider(minimum=0.0, maximum=1.0, value=0.6, step=0.01, label="Voxel Threshold")
                    steps_slider = gr.Slider(minimum=5, maximum=30, value=15, step=1, label="Number of Morphing Steps")
                
                generate_button = gr.Button("Generate Interpolation GIF", variant="primary")
                gif_output = gr.Image(label="Interpolation Animation", interactive=False)
                
                generate_button.click(
                    fn=self.interpolation_callback,
                    inputs=[
                        selected_item_A,
                        selected_item_B,
                        steps_slider,
                        threshold_slider_interp
                    ],
                    outputs=gif_output
                )
            # Tab 2: Reconstruction
            with gr.TabItem("🔄 Object Reconstruction"):
                gr.Markdown("<span style='font-family: Arial; font-size: 24px;'>"
                    "Select an object to Reconstruct. When you select an object from the gallery, " \
                "its voxel grid will be sent to the VAE for reconstruction.")
                with gr.Tabs() as category_tabs:
                    for category, previews in self.category_prews.items():
                        with gr.TabItem(category):
                            gr.Markdown(f"### {category.capitalize()} Objects")
                            gallery = gr.Gallery(
                                value=[p["image"] for p in previews],
                                label="Preview",
                                show_label=False,
                                columns=5,
                                height=360
                            )
                            with gr.Row():
                                #reconstruct_button = gr.Button("Generate Reconstruction", variant="primary")
                                threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.6, step=0.01, label="Threshold for the Voxel Grid")
                            with gr.Row():
                                original_plot_output = gr.Image(type="numpy", label="Original Object", interactive=False)
                                reconstructed_plot_output = gr.Image(type="numpy", label="VAE Reconstruction", interactive=False)
                            callback_for_this_tab = self.create_reconstruct_callback(previews)

                        
                            gallery.select(
                                fn=callback_for_this_tab,
                                inputs=[threshold_slider],
                                outputs=[original_plot_output, reconstructed_plot_output]
                            )



