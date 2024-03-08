---
layout: post
title: Jupyter Notebooks to Jekyll Blogs
categories: [jupyter notebook, jekyll, github pages]
---

## Background ##

I wanted to have a personal technical blog to share my learnings. While I initially used Github pages, I attempted to create everything from scratch, resulting in a blog that looked extremely crude. This led me to explore alternatives.

#### My crude looking blog! ####

![png]({{site.baseurl}}/assets/image/jupyter_notebook_to_jekyll_github_pages_files/earlier_blog.png "Earlier blog")


## Context ##

Recently, I discovered [Jekyll](https://jekyllrb.com/), a static site generator with built-in support for Github Pages. While exploring, I found [jekyll forever](https://github.com/forever-jekyll/forever-jekyll), a theme-template-boilerplate for Jekyll. Setting up a blog with it is remarkably simple and straightforward – justclone the repository and follow the steps outlined in the ReadMe file. Creating a new blog post is just as easy – write your content in markdown format, paste it into the _posts/ folder, commit, and voila! Your post appears on the blog instantly.

Jekyll supports markdown and HTML formats. However, in Machine Learning, where we often work with Jupyter notebooks, I faced the challenge of converting ipynb files into blog posts. I came across a helpful [blog post](https://jaketae.github.io/blog/jupyter-automation) that addressed this issue, getting almost everything right. However, there were a few issues and additional manual steps involved. To streamline the process, I rewrote the entire solution in Python. Now, the conversion from ipynb to a blog post can be accomplished with a single command. You can find the Python script [here](https://gist.github.com/chittiman/0ff85442c69dc5a9dc5db04b737a379b). Let me walk you through the code below.

## File Structure ##

Below is the file structure of the root folder. The notebooks intended for publication should be placed in the **_notebooks** folder. Upon running the script, corresponding markdown files will be generated and stored in the **_posts** folder. In case there are graphs in the notebook, a folder with the same file name will be created. This folder will then be moved to the **assets/image** directory.


```python
cur_dir = Path.cwd()
nb_dir = cur_dir / '_notebooks' # notebooks folder
md_dir = cur_dir / '_posts' # posts folder
imgs_dir = cur_dir / 'assets' / 'image' #images folder
```

![png]({{site.baseurl}}/assets/image/jupyter_notebook_to_jekyll_github_pages_files/file_structure.png "file_structure")


## Step 1 - Convert Notebooks to Markdown: ##

The **nbconvert** module, included with Jupyter Notebook, facilitates the conversion of .ipynb files to .md files while extracting graphs and images from the notebooks. These images and graphs are then stored in a separate folder.


```python
cmd = f"jupyter nbconvert --to markdown {args.file}"
subprocess.run(cmd, shell=True)
```

## Step 2 - Relocating Images to the Appropriate Location: ##

The extracted images, currently residing in the **_notebooks** folder, need to be moved to the **assets/image** folder. An if condition is in place to address cases where there are no graphs in the .ipynb file.


```python
notebook = Path(args.file)
file_prefix = notebook.name.replace('.ipynb', '')

#Moving images folder to right place
img_dir = nb_dir / f'{file_prefix}_files'
new_img_dir = imgs_dir / img_dir.name
if img_dir.exists():
    img_dir.rename(new_img_dir)
else:
    new_img_dir.mkdir(exist_ok=True)
```

## Step 3 - Standardizing Markdown File Names: ##

To comply with Jekyll's naming convention **(YYYY-MM-DD-text.md)**, I'm standardizing the file names. The text provided here becomes part of the blog post URL. Once the markdown text is cleaned, it is then written to the **_posts** folder.

**Ex**: 2021-07-15-hello-world.md \
**URL**: blog_name.github.io/hello-world


```python
md_file = nb_dir / f'{file_prefix}.md'
clean_md_name = re.sub(r'[\s_]', '-',md_file.name).lower()
#replacing space and _ with -
new_md_name = f'{args.date}-{clean_md_name}'
new_md_file = md_dir / new_md_name
```

## Step 4 - Adding Front Matter: ##

Jekyll requires front matter at the start of our markdown files. This metadata, including title and tags, follows a specific format. I've created a simple function to extract information from command line arguments and convert it into the required front matter.

![png]({{site.baseurl}}/assets/image/jupyter_notebook_to_jekyll_github_pages_files/front_matter_format.png "front_matter")



```python
def get_front_matter(args):
    cat_str = ', '.join(args.categories)
    layout = f"""
---
layout: post
title: {args.title}
categories: [{cat_str}]
---
"""
```

## Step 5 - Fixing Image Paths: ##


As we move both the image folder and markdown files, it's crucial to correct the image paths inside the markdown file. Using a simple regex function, we insert **{{site.baseurl}}/assets/image/** behind the previous path to ensure accurate referencing.


```python
text = re.sub(r'(!\[png\]\()(.*)\)',r'\1{{site.baseurl}}/assets/image/\2)' ,text)
#Correcting assets folder path
```

## Step 6 - Wrapping Up: ##

The cleaned text with added front matter is saved in a .md file within the 'posts' folder, concluding the process. The raw markdown file is then deleted.


```python
# Adding front matter at the start
front_matter = get_front_matter(args)
clean_text = front_matter + "\n\n" + text
write_file(tgt_file,clean_text)

#Deleting raw .md file
md_file.unlink()
```

## Step 7 - How to Execute: ##

Start by creating a **_notebooks** folder within the root directory and placing your ipynb file there. Ensure this folder is added to the exclude section in the 'config.yml' file.

[Link to Python script](https://gist.github.com/chittiman/0ff85442c69dc5a9dc5db04b737a379b): Place this script in the root folderand run the command in the following format from the root folder.


```python
python ./ipynb_to_md.py 
        -file ./_notebooks/test.ipynb
        -title "test post"
        -date "2024-01-02"
        -categories "testing" "experimenting"
```
