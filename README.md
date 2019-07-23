# art-render
A deep learning project using neural style transfer to render your images in the form of various art movements in history.

# Usage
Clone the entire repository of this project (especially the images!).

From the cloned directory, run the Python script art_transform.py with the following arguments :

python art_transform.py -s STYLE -i INPUT_IMG_PATH

  -s STYLE           Enter the preferred style of the output image : 1 for
                     Impressionism, 2 for Post-Impressionism, 3 for Cubism, 4
                     for Fauvism, 5 for Expressionism, 6 for Surrealism, 7 for
                     Romanticism, 8 for Abstract Expressionism, 9 for
                     Renaissance, 10 for Modern Art.
                     
  -i INPUT_IMG_PATH  The path to the input image, with its full name.
  
For example,
  
python art_transform.py -s 1 -i charminar.jpg
  
The output of the script will be a file called output.jpg containing the rendering of your input image in the desired artistic style.
