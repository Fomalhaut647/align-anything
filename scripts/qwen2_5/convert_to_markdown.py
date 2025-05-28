import json
import os

def convert_to_markdown():
    # Read the JSON file
    json_path = 'scripts/qwen2_5/evaluation_results/evaluation_results.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create markdown content
    markdown_content = "# Evaluation Results\n\n"
    
    for i, item in enumerate(data, 1):
        markdown_content += f"## Example {i}\n\n"
        markdown_content += "### Prompt\n"
        markdown_content += f"{item['prompt']}\n\n"
        
        markdown_content += "### Base Response\n"
        markdown_content += f"{item['base_response']}\n\n"
        
        markdown_content += "### DPO Response\n"
        markdown_content += f"{item['dpo_response']}\n\n"
        
        markdown_content += "---\n\n"  # Add separator between examples
    
    # Write to markdown file
    output_path = 'scripts/qwen2_5/evaluation_results/evaluation_results.md'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

if __name__ == "__main__":
    convert_to_markdown() 