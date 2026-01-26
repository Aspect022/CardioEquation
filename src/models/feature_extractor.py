import tensorflow as tf
from tensorflow.keras import layers, Model

class FeatureExtractor(Model):
    """
    Stage 1: Feature Extraction
    Ref: '1D ResNet-18 backbone' from CardioEquation 2.0 Spec (Part 3 & 9)
    Input: (Batch, 2500, 1) Noisy ECG
    Output: (Batch, 512) Feature Vector
    """
    def __init__(self, input_shape=(2500, 1)):
        super(FeatureExtractor, self).__init__()
        
        # Initial Conv
        self.conv1 = layers.Conv1D(64, kernel_size=15, strides=2, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.pool1 = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')
        
        # ResNet Blocks
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        
        self.global_pool = layers.GlobalAveragePooling1D()
        self.flatten = layers.Flatten()
        
    def _make_layer(self, filters, blocks, stride):
        return [ResBlock(filters, stride) for _ in range(blocks)]

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Unroll blocks
        for block in self.layer1: x = block(x)
        for block in self.layer2: x = block(x)
        for block in self.layer3: x = block(x)
        for block in self.layer4: x = block(x)
            
        x = self.global_pool(x)
        return self.flatten(x)

class ResBlock(layers.Layer):
    def __init__(self, filters, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = layers.Conv1D(filters, kernel_size=7, strides=stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv1D(filters, kernel_size=7, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        self.downsample = None
        if stride != 1:
            self.downsample = layers.Conv1D(filters, kernel_size=1, strides=stride, use_bias=False)
            self.ds_bn = layers.BatchNormalization()

    def call(self, inputs):
        residual = inputs
        
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(inputs)
            residual = self.ds_bn(residual)
            
        out += residual
        out = self.relu(out)
        return out
