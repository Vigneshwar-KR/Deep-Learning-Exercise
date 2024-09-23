import cv2
import tensorflow as tf
from tensorflow.keras.models import Model

class GradCAM:
  def __init__(self, model, class_id, layer_name=None):
    """Constructor

    Args:
        model (keras.Model): a Keras model created using
          the functional API.
        class_id (int): id of the class to compute the Grad CAM
        layer_name (string, optional): Name of the layer used to
          compute the Grad CAM. Defaults to None.
    """
    self.model = model    
    self.class_id = class_id
    self.layer_name = layer_name

    if self.layer_name is None:
      self.layer_name = self.find_target_layer()
    
    self.cam = None
    self.weights = None    

  def find_target_layer(self):
    """Find the last layer with an output shape
      equals to 4

    Returns:
        string: layer name
    """
    for layer in reversed(self.model.layers):
      if len(layer.output_shape) == 4:
        return layer.name
    
  def compute_heatmap(self, image, eps=1e-8):
    """Compute GradCAM

    Args:
        image (np.array): input image
        eps (float, optional): To avoid divisions
          by zero. Defaults to 1e-8.

    Returns:
        np.array: image with GradCAM as heatmap
    """
    # Model to get the output of the last convolutional layer
    grad_model = Model(inputs=[self.model.inputs],
                       outputs=[self.model.get_layer(self.layer_name).output,
                                self.model.output])
    
    # Run model and record outputs, loss and gradients
    # tf.GradientTape is an automatic differentiation engine which 
    # records all the derivatives within its scope
    with tf.GradientTape() as tape:
      # cast the image tensor to float32
      inputs = tf.cast(image, tf.float32)
      # pass the forward propagate the image through the gradient model
      # and grab the loss associated with the specific class index
      (conv_outputs, predictions) = grad_model(inputs)
      loss = predictions[:, self.class_id]

    # gradients of loss with respect to conv layer
    grads = tape.gradient(loss, conv_outputs)

    # guided backprop (elimination of negative values)
    cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
    cast_grads = tf.cast(grads > 0, "float32")
    guided_grads = cast_conv_outputs * cast_grads * grads

    # discard the batch dimension
    conv_outputs = conv_outputs[0]
    guided_grads = guided_grads[0]

    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    self.weights = tf.reduce_mean(guided_grads, 
                                  axis=(0,1))
    self.cam = tf.reduce_sum(tf.multiply(self.weights, 
                                         conv_outputs),
                             axis=-1)    
    
    # resize it to get a heatmap with the same size than the
    # input image
    (w, h) = (image.shape[2], image.shape[1])
    heatmap = cv2.resize(self.cam.numpy(), (w,h))
    
    # normalize between [0,1], scale to the range [0,255]
    # convert to unsigned 8-bit integer
    numer = heatmap - heatmap.min()
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")

    return heatmap

  def overlay_heatmap(self, 
                      heatmap, 
                      image, 
                      alpha=0.5, 
                      colormap=cv2.COLORMAP_VIRIDIS):
    """Overlay the GradCAM's heatmap to the input image

    Args:
        heatmap (np.array): GradCAM's heatmap
        image (np.array): input image
        alpha (float, optional): Weight use to overlay
          the heatmap to the input image. Defaults to 0.5.
        colormap (np.array, optional): Colormap. Defaults
          to cv2.COLORMAP_VIRIDIS.

    Returns:
        (np.array, np.array): tuple with the heatmap and
          the image with the heatmap overlayed
    """
    
    heatmap = cv2.applyColorMap(heatmap, colormap)
    # output =  alpha*image + (1-alpha)*heatmap
    output = cv2.addWeighted(image, 
                             alpha, 
                             heatmap, 
                             1-alpha, 
                             0)
    return (heatmap, output)