
# coding: utf-8

# # Deep Learning & Art: Neural Style Transfer
# 

# In[1]:


import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

get_ipython().magic(u'matplotlib inline')

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
print(model)
# In[3]:


content_image = scipy.misc.imread("images/louvre.jpg")
imshow(content_image)


# In[4]:




def compute_content_cost(content_C, content_G):
    """
    Computes the content cost
    
    Arguments:
    content_C -- matrix of dimension (1, n_H, n_W, n_C), content of the "content" image
    content_G -- tensor of dimension (1, n_H, n_W, n_C), content of the "generated" image
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
 
    # Retrieve dimensions from content_G
    m, n_H, n_W, n_C = content_G.get_shape().as_list()
    
    # Reshape content_C and content_G
    content_C_unrolled = tf.transpose(tf.reshape(content_C, [n_H*n_W, n_C]))
    content_G_unrolled = tf.transpose(tf.reshape(content_G, [n_H*n_W, n_C]))
    
    # compute the cost with tensorflow
    J_content = (1 / (4 * n_H * n_W * n_C)) * tf.reduce_sum(tf.square(tf.subtract(content_C_unrolled, content_G_unrolled)))
    
    
    return J_content


# In[5]:


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    content_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    content_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(content_C, content_G)
    print("J_content = " + str(J_content.eval()))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **J_content**
#         </td>
#         <td>
#            6.76559
#         </td>
#     </tr>
# 
# </table>

# In[6]:


style_image = scipy.misc.imread("images/monet_800600.jpg")
imshow(style_image)


# In[7]:




def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
   
    GA = tf.matmul(A, tf.transpose(A))
  
    
    return GA


# In[8]:


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2*1], mean=1, stddev=4)
    GA = gram_matrix(A)
    
    print("GA = " + str(GA.eval()))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **GA**
#         </td>
#         <td>
#            [[  6.42230511  -4.42912197  -2.09668207]
#  [ -4.42912197  19.46583748  19.56387138]
#  [ -2.09668207  19.56387138  20.6864624 ]]
#         </td>
#     </tr>
# 
# </table>

# In[9]:




def compute_style_cost(style_S, style_G):
    """
    Arguments:
    style_image -- a matrix of dimension (m, n_H, n_W, n_C)
    generated_image -- a matrix of dimension (Ho, Wo, n_C)
    
    Returns: 
    J_style -- scalar, style cost defined above by equation (2)
    """
    
  
    # Retrieve dimensions from style_G
    m, n_H, n_W, n_C = style_G.get_shape().as_list()
    
    # Reshape the images to have them of shape(F, Ho*Wo).
    style_S = tf.transpose(tf.reshape(style_S, [n_H*n_W, n_C]))
    style_G = tf.transpose(tf.reshape(style_G, [n_H*n_W, n_C]))

    # Computing gram_matrices
    GS = gram_matrix(style_S)
    GG = gram_matrix(style_G)

    # Computing the loss
    J_style = 1/(4 * n_C**2 * (n_H*n_W)**2) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))
    
   
    return J_style


# In[10]:


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    style_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    style_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style = compute_style_cost(style_S, style_G)
    
    print("J_style = " + str(J_style.eval()))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **J_style**
#         </td>
#         <td>
#            9.19028
#         </td>
#     </tr>
# 
# </table>

# In[11]:


STYLE_LAYERS = [
    ('conv1_1', 0.5),
    ('conv2_1', 1.0),
    ('conv3_1', 1.5),
    ('conv4_1', 3.0),
    ('conv5_1', 4.0)]


# In[12]:




def total_cost(J_content, J_style, alpha = 5, beta = 100):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    
 
    J = alpha * J_content + beta * J_style

    
    return J


# In[13]:


tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
        
    content_A = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    content_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(content_A, content_C)
    
    style_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    style_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style = compute_style_cost(style_S, style_G)
    J = total_cost(J_content, J_style)
    
    print("J = " + str(J.eval()))


# **Expected Output**:
# 
# <table>
#     <tr>
#         <td>
#             **J**
#         </td>
#         <td>
#            736.912
#         </td>
#     </tr>
# 
# </table>

# In[14]:


# Reset the graph
tf.reset_default_graph()

# Start interactive session

sess = tf.InteractiveSession()


# In[15]:


content_image = scipy.misc.imread("images/louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)


# In[16]:


style_image = scipy.misc.imread("images/monet.jpg")
style_image = reshape_and_normalize_image(style_image)


# In[17]:


generated_image = generate_noise_image(content_image)
imshow(generated_image[0])


# As explained in part (2), we need to load the VGG16 model.

# In[18]:


model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")


# In[19]:


# Assign the input of the model to be the "content" image 
sess.run(model['input'].assign(content_image))

# Select the output tensor of the 4th convolutional layer
out = model['conv4_2']

# get the content from the content image by running the model with the content image assigned as input.
content_C = sess.run(out)

# get the content from the generated image
content_G = out

# Compute the content cost
J_content = compute_content_cost(content_C, content_G)


# In[20]:


# Assign the input of the model to be the "style" image 

sess.run(model['input'].assign(style_image))


# initialize the style cost
J_style = 0

for layer_name, coeff in STYLE_LAYERS:
    
  
    # Select the output tensor of the current layer
    out = model[layer_name]
    
    # Get the style of the style image by running the session on out
    style_S = sess.run(out)
    
    # Get the style of the generated image. It's the output tensor of the current layer.
    style_G = out
    
    # Compute style_cost for the current layer
    style_cost = compute_style_cost(style_S, style_G)
    
    # Add style_cost*coeff of this layer to overall style cost
    J_style = J_style + coeff * style_cost
    
  


# In[23]:



J = total_cost(J_content, J_style, alpha = 10, beta = 100)


# In[25]:


def model_nn(sess, input_image, num_iterations = 1500):
    
    # Initialize global variables
 
    sess.run(tf.global_variables_initializer())
 
    
    # Run the noisy input image (initial generated image) through the model. Use assign().
  
    sess.run(model['input'].assign(input_image))
   
    
    for i in range(num_iterations):
    
        # Run the session on the train_step to minimize the total cost
       
        sess.run(train_step)
     
        # Compute the generated image by running the session on the current model['input']
       
        generated_image = sess.run(model['input'])
    
        # Print every 100 iteration.
        if i%100 == 0:
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(sess.run(J)))
            print("content cost = " + str(sess.run(J_content)))
            print("style cost = " + str(sess.run(J_style)))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image

