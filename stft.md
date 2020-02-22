# STFT

## torch

$$
stft = real - j * imag
$$

```Python
stft = torch.stft(*args, **kwargs)
real = stft[..., 0]
imag = stft[..., 1]

mag = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)  # (..., frames, fft_size // 2 + 1)
```

## tensorflow

$$
stft = mag * exp(-j * angel) 
$$

```Python
stfts = tf.contrib.signal.stft(*args, **kwargs)  
mag = tf.abs(stfts)  
angle = tf.angle(stfts)  

stft = tf.multiply(tf.complex(mag, 0.0), tf.exp(tf.complex(0.0, angle)))  # (..., frames, fft_size // 2 + 1)
```
