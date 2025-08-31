"""
Custom nbconvert configuration for clean slideshow export
Hides code cells and optimizes for presentation
"""

from nbconvert import SlidesExporter
from traitlets.config import Config

def get_slides_config():
    """Configure nbconvert for clean slide export"""
    c = Config()
    
    # Hide input code cells
    c.TagRemovePreprocessor.enabled = True
    c.TagRemovePreprocessor.remove_cell_tags = {"hide_cell"}
    c.TagRemovePreprocessor.remove_input_tags = {"hide_input"}
    c.TagRemovePreprocessor.remove_output_tags = {"hide_output"}
    
    # Configure slides exporter
    c.SlidesExporter.exclude_input = True
    c.SlidesExporter.exclude_output_prompt = True
    c.SlidesExporter.exclude_input_prompt = True
    
    # Use reveal.js for better presentations
    c.SlidesExporter.reveal_theme = 'white'
    c.SlidesExporter.reveal_transition = 'slide'
    c.SlidesExporter.reveal_scroll = True
    
    return c

def create_clean_slides(input_file, output_file):
    """Convert notebook to clean slides without code"""
    config = get_slides_config()
    exporter = SlidesExporter(config=config)
    
    (body, resources) = exporter.from_filename(input_file)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(body)
    
    print(f"Clean slides exported to: {output_file}")

if __name__ == "__main__":
    create_clean_slides("slides.ipynb", "clean_slides.slides.html")