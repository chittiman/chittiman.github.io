from pathlib import Path
import subprocess
import argparse
import re

cur_dir = Path.cwd()
nb_dir = cur_dir / '_notebooks' # notebooks dir
md_dir = cur_dir / '_posts'
imgs_dir = cur_dir / 'assets' / 'image'

def convert_nb_md(args):
    cmd = f"jupyter nbconvert --to markdown {args.file}"
    subprocess.run(cmd, shell=True)
    notebook = Path(args.file)
    
    file_prefix = notebook.name.replace('.ipynb', '')

    img_dir = nb_dir / f'{file_prefix}_files'
    new_img_dir = imgs_dir / img_dir.name
    img_dir.rename(new_img_dir)

    md_file = nb_dir / f'{file_prefix}.md'
    clean_md_name = re.sub(r'[\s_]', '-',md_file.name).lower()#replacing space and _ with -
    new_md_name = f'{args.date}-{clean_md_name}'
    new_md_file = md_dir / new_md_name
    clean_md(md_file,new_md_file,args)
    md_file.unlink()
    #md_file.rename(new_md_file)


def clean_md(src_file,tgt_file,args):
    text = load_file(src_file)
    text = re.sub(r'(!\[png\]\()(.*)\)',r'\1{{site.baseurl}}/assets/image/\2)' ,text)
    #text = re.sub(r'(!\[png\]\()(.*)\)',r'<img src="/assets/image/\2"' ,text)

    #<img src="/assets/images/some_file_name.png">
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

# def clean_md_file(md_file):
#     lines = load_file(md_file)
    
    




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
#     text = load_file(args.file)

# #![png](Introduction_to_Weight_Quantization_files/Introduction_to_Weight_Quantization_12_0.png)
#     #matches = re.finditer(r'(!\[png\]\()(.*)\)', text)
#     path = 'assets/image/'
#     print(re.sub(r'(!\[png\]\()(.*)\)',r'\1assets/image/\2' ,text))
    # print(type(matches[0]))
    # for match in matches:
    #     print(match.groups())
    # # #print(get_front_matter(args))
    #print(args.title)
    #all_cats = ', '.join(args.categories)
    #print(f'[{all_cats}]')

    #'[%s]' % ', '.join(map(str, args.categories))
    convert_nb_md(args)
    #print(load_file(args.file))
