import tensorflow as tf
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt

def unet(input_size=(256, 256, 3), num_filters=64):
    inputs = tf.keras.layers.Input(input_size)

    
    conv1 = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(num_filters*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(num_filters*2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(num_filters*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(num_filters*4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(num_filters*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(num_filters*8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(num_filters*16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(num_filters*16, (3, 3), activation='relu', padding='same')(conv5)

    up6 = tf.keras.layers.Conv2DTranspose(num_filters*8, (2, 2), strides=(2, 2), padding='same')(conv5)
    merge6 = tf.keras.layers.concatenate([conv4, up6])
    conv6 = tf.keras.layers.Conv2D(num_filters*8, (3, 3), activation='relu', padding='same')(merge6)
    conv6 = tf.keras.layers.Conv2D(num_filters*8, (3, 3), activation='relu', padding='same')(conv6)

    up7 = tf.keras.layers.Conv2DTranspose(num_filters*4, (2, 2), strides=(2, 2), padding='same')(conv6)
    merge7 = tf.keras.layers.concatenate([conv3, up7])
    conv7 = tf.keras.layers.Conv2D(num_filters*4, (3, 3), activation='relu', padding='same')(merge7)
    conv7 = tf.keras.layers.Conv2D(num_filters*4, (3, 3), activation='relu', padding='same')(conv7)

    up8 = tf.keras.layers.Conv2DTranspose(num_filters*2, (2, 2), strides=(2, 2), padding='same')(conv7)
    merge8 = tf.keras.layers.concatenate([conv2, up8])
    conv8 = tf.keras.layers.Conv2D(num_filters*2, (3, 3), activation='relu', padding='same')(merge8)
    conv8 = tf.keras.layers.Conv2D(num_filters*2, (3, 3), activation='relu', padding='same')(conv8)

    up9 = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(conv8)
    merge9 = tf.keras.layers.concatenate([conv1, up9])
    conv9 = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(merge9)
    conv9 = tf.keras.layers.Conv2D(num_filters, (3, 3), activation='relu', padding='same')(conv9)

    outputs = tf.keras.layers.Conv2D(3, (1, 1), activation='sigmoid', padding='same')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def tryon(input_size=(256, 256, 3)):
    pessoa_entrada = tf.keras.layers.Input(input_size)
    roupa_entrada = tf.keras.layers.Input(input_size)

    unet_dados = unet(input_size=input_size)
    unet_reconstrucao = unet(input_size=(input_size[0], input_size[1], 6))

    dados_pessoa = unet_dados(pessoa_entrada)
    dados_roupa = unet_dados(roupa_entrada)

    dados_combinados = tf.keras.layers.Concatenate(axis=3)([dados_pessoa, dados_roupa])

    tryon_output = unet_reconstrucao(dados_combinados)

    model = tf.keras.Model(inputs=[pessoa_entrada, roupa_entrada], outputs=tryon_output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

tryon_net = tryon(input_size=(256, 256, 3))

def carrega_imagem(caminho_imagem, target_size=(256, 256)):
    imagem = imread(caminho_imagem)
    imagem = resize(imagem, target_size, mode='constant', preserve_range=True)
    imagem = imagem / 255.0
    return imagem

caminho_pessoa = 'pessoa.png'
caminho_roupa = 'roupa.png'

imagem_pessoa = carrega_imagem(caminho_pessoa)
imagem_roupa = carrega_imagem(caminho_roupa)


plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(imagem_pessoa)
plt.title('Imagem Pessoa')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(imagem_roupa)
plt.title('Imagem Roupa')
plt.axis('off')

plt.show()

imagem_pessoa = np.expand_dims(imagem_pessoa, axis=0)
imagem_roupa = np.expand_dims(imagem_roupa, axis=0)

resultado_tryon = tryon_net.predict([imagem_pessoa, imagem_roupa])

imagem_resultado = np.clip(resultado_tryon[0], 0, 1)

plt.imshow(imagem_resultado)
plt.axis('off') 
plt.show()
