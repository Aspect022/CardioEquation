import tensorflow as tf
from tensorflow.keras import layers, Model

class ConditionalDiffusionUNet(Model):
    """
    Stage 2: Conditional Diffusion Denoiser (CORE REPLACEMENT)
    Architecture: U-Net with time embedding + conditioning
    Ref: Part 3, Stage 2
    """
    def __init__(self, input_shape=(2500, 1), condition_dim=512):
        super(ConditionalDiffusionUNet, self).__init__()
        
        # Time Embedding MLP
        self.time_mlp = tf.keras.Sequential([
            layers.Dense(256 * 4, activation='relu'),
            layers.Dense(256 * 4) # Time embedding dim
        ])
        
        # DOWN BLOCKS
        self.conv_in = layers.Conv1D(64, 3, padding='same')
        self.down1 = UnetBlock(64, 128)
        self.down2 = UnetBlock(128, 256)
        self.down3 = UnetBlock(256, 512)
        
        # BOTTLENECK with Conditioning
        self.bottleneck1 = UnetBlock(512, 1024)
        
        # UP BLOCKS
        self.up3 = UnetBlock(1024 + 512, 512, up=True) # Skip input size
        self.up2 = UnetBlock(512 + 256, 256, up=True)
        self.up1 = UnetBlock(256 + 128, 128, up=True)
        
        self.conv_out = layers.Conv1D(1, 3, padding='same')
        
    def call(self, inputs, timestep, conditioning):
        # inputs: (B, 2500, 1) Noisy Signal
        # timestep: (B, ) Noise level
        # conditioning: (B, 512) Feature Vector
        
        # Embed Time
        t_emb = self.time_mlp(tf.expand_dims(timestep, -1)) # (B, 1024)
        
        # Condition Injection (Concatenate to time or intermediate?)
        # EDDM Approach: Concat condition to bottleneck or add to time emb
        # Let's project condition to same dim as time and Add
        cond_emb = layers.Dense(1024)(conditioning)
        global_cond = t_emb + cond_emb
        
        # UNet Pass
        x1 = self.conv_in(inputs)    # 2500, 64
        x2 = self.down1(x1)          # 1250, 128
        x3 = self.down2(x2)          # 625, 256
        x4 = self.down3(x3)          # 312, 512 (approx)
        
        # Inject Condition into Bottleneck
        # Reshape global_cond: (B, 1, 1024)
        cond_reshaped = tf.expand_dims(global_cond, 1)
        
        x_bottle = self.bottleneck1(x4) # 156, 1024
        
        # Add global info (broadcasting)
        x_bottle = x_bottle + cond_reshaped 
        
        # Decoder
        x = self.up3(x_bottle, x4)
        x = self.up2(x, x3)
        x = self.up1(x, x2)
        
        # Final Upsample to original size (if needed)
        x = layers.UpSampling1D(size=2)(x) # 2500
        x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
        output = self.conv_out(x)
        
        return output

class UnetBlock(layers.Layer):
    def __init__(self, in_ch, out_ch, up=False):
        super(UnetBlock, self).__init__()
        self.up = up
        if up:
            self.upsample = layers.UpSampling1D(size=2)
            self.conv = layers.Conv1D(out_ch, 3, padding='same')
        else:
            self.conv = layers.Conv1D(out_ch, 3, strides=2, padding='same')
            
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation('relu')
        self.conv2 = layers.Conv1D(out_ch, 3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.Activation('relu')

    def call(self, x, skip=None):
        if self.up:
            x = self.upsample(x)
            # Handle shape mismatch due to odd padding
            if skip is not None:
                # Crop or Pad? Let's assume input is power of 2 roughly or resize
                # Resize x to match skip
                target_len = tf.shape(skip)[1]
                x = tf.image.resize(tf.expand_dims(x, -1), [target_len, 1]) # Hack for 1D resize via 2D
                x = tf.squeeze(x, -1)
                x = tf.concat([x, skip], axis=-1)
                
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x
