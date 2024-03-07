from pathlib import Path
import subprocess
import argparse
import re

cur_dir = Path.cwd()
nb_dir = cur_dir / '_notebooks' # notebooks folder
md_dir = cur_dir / '_posts' # posts folder
imgs_dir = cur_dir / 'assets' / 'image' #images folder

def convert_nb_md(args):
    #Notebook conversion to markdown
    #Creates .md file and folder with assets used in notebook
    cmd = f"jupyter nbconvert --to markdown {args.file}"
    subprocess.run(cmd, shell=True)
    notebook = Path(args.file)
    
    file_prefix = notebook.name.replace('.ipynb', '')

    #Moving images folder to right place
    img_dir = nb_dir / f'{file_prefix}_files'
    new_img_dir = imgs_dir / img_dir.name
    if img_dir.exists():
        img_dir.rename(new_img_dir)
    else:
        new_img_dir.mkdir(exist_ok=True)

    md_file = nb_dir / f'{file_prefix}.md'
    clean_md_name = re.sub(r'[\s_]', '-',md_file.name).lower()
    #replacing space and _ with -
    new_md_name = f'{args.date}-{clean_md_name}'
    new_md_file = md_dir / new_md_name

    #Cleaning markdown file
    clean_md(md_file,new_md_file,args)
    #Deleting raw .md file
    md_file.unlink()
    #md_file.rename(new_md_file)


def clean_md(src_file,tgt_file,args):
    text = load_file(src_file)
    text = re.sub(r'(!\[png\]\()(.*)\)',r'\1{{site.baseurl}}/assets/image/\2)' ,text)
    #Correcting assets folder path

    # Adding front matter at the start
    front_matter = get_front_matter(args)
    clean_text = front_matter + "\n\n" + text
    write_file(tgt_file,clean_text)

def load_file(file):
    with open(file,'r',encoding='utf-8') as f:
        text = f.read()
    return text

def write_file(file,text):
    with open(file, 'w+', encoding='utf-8') as f:
        f.write(text)

def get_front_matter(args):
    cat_str = ', '.join(args.categories)
    layout = f"""
---
layout: post
title: {args.title}
categories: [{cat_str}]
---
"""
    return layout.strip()

    

if __name__== '__main__':
    parser = argparse.ArgumentParser(description='jupyter notebook to markdown')
    parser.add_argument('-file', type=str, help='Path to the file')
    parser.add_argument('-date', type=str, help='date prefix')
    parser.add_argument('-title', type=str, help='title')
    parser.add_argument('-categories', nargs='*')
    args = parser.parse_args()
    convert_nb_md(args)
    #Sample cmd 
    #python ./nb_to_md.py -file ./_notebooks/test.ipynb -title "test post" -date "2024-01-02" -categories "testing" "experimenting"
