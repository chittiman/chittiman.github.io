---
layout: post
title: Jupyter Notebooks to Jekyll Blogs
categories: [jupyter notebook, jekyll, github pages]
---

# Convert Jupyter notebooks to Jekyll Blog Posts

## Background ##

I wanted to have a personal technical blog where I can publish my learnings. I used Github pages earlier but I tried to do create everything from scratch. I was left with a blog which looked extremely crude.So, I was in search of alternatives.

### My crude looking blog! ###

![png]({{site.baseurl}}/assets/image/jupyter_notebook_to_jekyll_github_pages_files/earlier_blog.png "Earlier blog")


## Context ##

Recently, I found out about [jekyll forever](https://github.com/forever-jekyll/forever-jekyll) and the process of setting up a blog is really simple and straightforward. You just have to clone the repo and follow the steps mentioned in the ReadMe file. To create a new blog post, you just have to write down the blog in markdown format and paste it in the _posts/ folder and push commit. Your post appears on the blog instantly. But, jekyll supports markdown and html formats. Since, in Machine Learning we deal with jupyter notebooks day-to-day, I needed to figure out a way to  convert ipynb files into blog posts.

I have found a [blog](https://jaketae.github.io/blog/jupyter-automation) which solves the issue. It gets almost all things right except for a few issues and some additional manual steps. I have rewritten everything in Python, so that the conversion from ipynb to blog post can happen in a single command. You can find the python script [here](https://gist.github.com/chittiman/0ff85442c69dc5a9dc5db04b737a379b). I will explain the code below

## File Structure ##

Present below is the file structure of the root folder. The notebooks we want to publish have to be placed in notebooks folder. Once we run the script, corresponding markdown files will appear in posts folder. If any images or graphs are present in the notebook, they will be stored in a folder with same file name. And this folder finally will be shifted to assets/image folder


```python
cur_dir = Path.cwd()
nb_dir = cur_dir / '_notebooks' # notebooks folder
md_dir = cur_dir / '_posts' # posts folder
imgs_dir = cur_dir / 'assets' / 'image' #images folder
```

![png]({{site.baseurl}}/assets/image/jupyter_notebook_to_jekyll_github_pages_files/file_structure.png "file_structure")


## Step 1 - Notebook to markdown conversion ##

For this, nbconvert module which comes bundled with jupyter notebook is handy. It converts .ipynb files to .md files and extracts images/graphs in notebooks and stores them in separate folder.


```python
cmd = f"jupyter nbconvert --to markdown {args.file}"
subprocess.run(cmd, shell=True)
```

## Step 2 - Moving images to appropriate location ##

Extracted images which are currently in the notebooks folder have to be shifted to assets/image folder. If condition exists for a case where there are no images in ipynb file.


```python
notebook = Path(args.file)
file_prefix = notebook.name.replace('.ipynb', '')

#Moving images folder to right place
img_dir = nb_dir / f'{file_prefix}_files'
new_img_dir = imgs_dir / img_dir.name
if img_dir.exists():
    img_dir.rename(new_img_dir)
```

## Step 3 - Moving images to appropriate location ##

Jekyll follows a strict naming convention for the markdown file. File name has to start with date in YYYY-MM-DD format followed by text. So, I am standardizing the file name.This text will be in the url of the blogpost. Since my markdown file finally has to be in posts folder,I'm creating a new path so that I can write the cleaned markdown text there. \
Ex: 2021-07-15-hello-world.md


```python
md_file = nb_dir / f'{file_prefix}.md'
clean_md_name = re.sub(r'[\s_]', '-',md_file.name).lower()
#replacing space and _ with -
new_md_name = f'{args.date}-{clean_md_name}'
new_md_file = md_dir / new_md_name
```

## Step 4 - Creating the front matter ##

At the start of markdown file we want to publish,jekyll expects frontmatter. Frontmatter is metadata like title and tagshas in a particular format. I wrote a simple function to extract the relevant information from cmd line arguments and convert them into front matter.

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

## Step 5 - Correcting image paths inside ##


Since we are going to move the image folder and markdown files, the image paths inside the markdown file have to point to new location. We use simple regex function to correct these paths. \
**{{site.baseurl}}/assets/image/** is inserted behind the earlier path to correct it.


```python
text = re.sub(r'(!\[png\]\()(.*)\)',r'\1{{site.baseurl}}/assets/image/\2)' ,text)
#Correcting assets folder path
```

## Step 6 - Tying up all loose ends ##

Front matter is added at the start to this cleaned text and this text is written in .md file in posts folder. And the raw markdown file is deleted. 


```python
# Adding front matter at the start
front_matter = get_front_matter(args)
clean_text = front_matter + "\n\n" + text
write_file(tgt_file,clean_text)

#Deleting raw .md file
md_file.unlink()
```

## Step 7 - How to run ?? ##

First create **_notebooks** folder inside the root directory and place your ipynb file there. Add this folder to exclude section in config.yml file. \

Place this [python script](https://gist.github.com/chittiman/0ff85442c69dc5a9dc5db04b737a379b) inside the root folder. Make sure nbconvert is already installed. Run the cmd in the below format from root folder. 


```python
python ./ipynb_to_md.py 
        -file ./_notebooks/test.ipynb
        -title "test post"
        -date "2024-01-02"
        -categories "testing" "experimenting"
```
