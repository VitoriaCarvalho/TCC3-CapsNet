import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";  # The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"]="1";  # Do other imports now...

print('\n<< GPU OK >>\n')

print('\n<< Carregando libs... >>\n')

import keras.backend as K
import tensorflow as tf
from keras import initializers, layers

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, roc_curve, auc, roc_auc_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import LabelBinarizer
from glob import glob
import matplotlib.pyplot as plt
from skimage.io import imread_collection, imsave
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from skimage.transform import resize
import seaborn as sn
import pandas as pd
from skimage.exposure import equalize_hist

print('\n<< Libs OK >>\n')

def Norm(imgs=None, dim=128, channel="rgb", preprocessing=False):

    """
    Essa função é responsável por redimensionar as imagens para o tamanho definido e utilizar um canal específico
    :param imgs: lista de imagens
    :param dim: dimensão que as imagens devem ter, o padrão é 128x128
    :param channel: canal escolhido para as imagens, o padrão é rgb
    :param preprocessing: flag que define se as imagens passarão ou não por um pré-processamento com a equalização de histograma
    :return: array numpy com todas as imagens, dimensão final: (qtd_imagens,dim,dim,n_channels)
    """

    print("\n<< Função controle >>\n")
    print(f"<< Normalizando {len(imgs)} imagens para a dimensão {dim}x{dim} e canal {channel} >>\n")      
    list_imgs = []
    n_channels = 1
    
    for id_img, img in enumerate(imgs):
        
        # Histogram equalization
        if preprocessing:
            img = equalize_hist(img, nbins=256, mask=None)
       
        # Split channels
        if channel == "rgb":
            n_channels = 3
            
        elif channel == "gray":
            img = rgb2gray(img)
            
        elif channel == "red":
            img = img[:,:,0]
            
        elif channel == "blue":                   
            img = img[:,:,1]
            
        elif channel == "green":
            img = img[:,:,2]
            
        # Resize image
        img = Image.fromarray((img * 255).astype(np.uint8))
        img = img.resize((dim,dim), Image.ANTIALIAS)
        img = np.asarray(img)
        # img = resize(img, (dim, dim, 3)) if n_channels == 1 else resize(img, (dim, dim))
        list_imgs.append(img/np.max(img))
        
        if id_img == 0:
            print(img.shape)
            imsave('example_img.png', img)

    return np.asarray(list_imgs,dtype=np.float32).reshape(-1,dim,dim,n_channels)

print('\n<< Carregando imagens... >>\n')

glaucoma_imgs = imread_collection("./doente/*.png")
normal_imgs = imread_collection("./normal/*.png")

doente = Norm(imgs=glaucoma_imgs, dim=64, channel="rgb", preprocessing=True)
normal = Norm(imgs=normal_imgs, dim=64, channel="rgb", preprocessing=True)

data = np.concatenate((doente, normal), axis=0)
label = to_categorical(np.concatenate(([1]*len(doente),[0]*len(normal)),axis=0))

print('\n<< Imagens OK >>\n')

print('\n<< Dividindo conjunto em treino e teste... >>\n')
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.20, shuffle=True, random_state=42)
X_val = X_train[:55]
y_val = y_train[:55]
X_train = X_train[55:]
y_train = y_train[55:]

print(len(X_train), len(X_test), len(X_val))

print('\n<< Split OK >>\n')

print('\n<< Definindo funções da CapsNet... >>\n')

class Length(layers.Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def __init__(self, **kwargs):
        super(Length, self).__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1))

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]


class Mask(layers.Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)
    
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])


def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


class CapsuleLayer(layers.Layer):
    """
    The capsule layer. It is similar to Dense layer. Dense layer has `in_num` inputs, each is a scalar, the output of the 
    neuron from the former layer, and it has `out_num` output neurons. CapsuleLayer just expand the output of the neuron
    from scalar to vector. So its input shape = [None, input_num_capsule, input_dim_capsule] and output shape = \
    [None, num_capsule, dim_capsule]. For Dense Layer, input_dim_capsule = dim_capsule = 1.
    
    :param num_capsule: number of capsules in this layer
    :param dim_capsule: dimension of the output vectors of the capsules in this layer
    :param routings: number of iterations for the routing algorithm
    """
    def __init__(self, num_capsule, dim_capsule, routings=3,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        assert len(input_shape) >= 3, "The input Tensor should have shape=[None, input_num_capsule, input_dim_capsule]"
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        # Transform matrix
        self.W = self.add_weight(shape=[self.num_capsule, self.input_num_capsule,
                                        self.dim_capsule, self.input_dim_capsule],
                                 initializer=self.kernel_initializer,
                                 name='W')

        self.built = True

    def call(self, inputs, training=None):
        # inputs.shape=[None, input_num_capsule, input_dim_capsule]
        # inputs_expand.shape=[None, 1, input_num_capsule, input_dim_capsule]
        #inputs_expand = K.expand_dims(inputs, 1)
        inputs_expand = tf.expand_dims(inputs, 1)

        # Replicate num_capsule dimension to prepare being multiplied by W
        # inputs_tiled.shape=[None, num_capsule, input_num_capsule, input_dim_capsule]
        #inputs_tiled = K.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_tiled  = tf.tile(inputs_expand, [1, self.num_capsule, 1, 1])
        inputs_tiled  = tf.expand_dims(inputs_tiled, 4)

        # Compute `inputs * W` by scanning inputs_tiled on dimension 0.
        # x.shape=[num_capsule, input_num_capsule, input_dim_capsule]
        # W.shape=[num_capsule, input_num_capsule, dim_capsule, input_dim_capsule]
        # Regard the first two dimensions as `batch` dimension,
        # then matmul: [input_dim_capsule] x [dim_capsule, input_dim_capsule]^T -> [dim_capsule].
        # inputs_hat.shape = [None, num_capsule, input_num_capsule, dim_capsule]
        #inputs_hat = K.map_fn(lambda x: K.batch_dot(x, self.W, [2, 3]), elems=inputs_tiled)
        inputs_hat = tf.map_fn(lambda x: tf.matmul(self.W, x), elems=inputs_tiled) 

        # Begin: Routing algorithm ---------------------------------------------------------------------#
        # The prior for coupling coefficient, initialized as zeros.
        # b.shape = [None, self.num_capsule, self.input_num_capsule].
        #b = tf.zeros(shape=[K.shape(inputs_hat)[0], self.num_capsule, self.input_num_capsule])
        b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.num_capsule, 
                      self.input_num_capsule, 1, 1])

        assert self.routings > 0, 'The routings should be > 0.'
        for i in range(self.routings):
            # c.shape=[batch_size, num_capsule, input_num_capsule]
            #c = tf.nn.softmax(b, axis=1)
            c = layers.Softmax(axis=1)(b)
            print(c.shape)

            # c.shape =  [batch_size, num_capsule, input_num_capsule]
            # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
            # The first two dimensions as `batch` dimension,
            # then matmal: [input_num_capsule] x [input_num_capsule, dim_capsule] -> [dim_capsule].
            # outputs.shape=[None, num_capsule, dim_capsule]
            #outputs = squash(K.batch_dot(c, inputs_hat, [2, 2]))  # [None, 10, 16]
            outputs = tf.multiply(c, inputs_hat)
            outputs = tf.reduce_sum(outputs, axis=2, keepdims=True)
            outputs = squash(outputs, axis=-2)  # [None, 10, 1, 16, 1]

            if i < self.routings - 1:
                # outputs.shape =  [None, num_capsule, dim_capsule]
                # inputs_hat.shape=[None, num_capsule, input_num_capsule, dim_capsule]
                # The first two dimensions as `batch` dimension,
                # then matmal: [dim_capsule] x [input_num_capsule, dim_capsule]^T -> [input_num_capsule].
                # b.shape=[batch_size, num_capsule, input_num_capsule]
                #b += K.batch_dot(outputs, inputs_hat, [2, 3])
                outputs_tiled = tf.tile(outputs, [1, 1, self.input_num_capsule, 1, 1])
                agreement = tf.matmul(inputs_hat, outputs_tiled, transpose_a=True)
                b = tf.add(b, agreement)
        # End: Routing algorithm -----------------------------------------------------------------------#

        outputs = tf.squeeze(outputs, [2, 4])
        return outputs

    def compute_output_shape(self, input_shape):
        return tuple([None, self.num_capsule, self.dim_capsule])


def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
    Apply Conv2D `n_channels` times and concatenate all capsules
    :param inputs: 4D tensor, shape=[None, width, height, channels]
    :param dim_capsule: the dim of the output vector of capsule
    :param n_channels: the number of types of capsules
    :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)


def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

    # Shared Decoder model in training and prediction
    
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
    
    

    # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])

    # manipulate model
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model, decoder


def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


print('\n<< Funções da CapsNet OK >>\n')

print('\n<< Carregando o modelo... >>\n')

model, eval_model, manipulate_model, decoder_model = CapsNet(input_shape=X_train.shape[1:], 
                                              n_class=len(np.unique(np.argmax(y_train, 1))),
                                              routings=8)

model.summary()
decoder_model.summary()

print('\n<< Modelo carregado >>\n')

print('\n<< Compilando o modelo... >>\n')

model.compile(optimizer=optimizers.Adam(lr=0.0001),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1.,0.392],
                  metrics={'capsnet': 'accuracy'})

print('\n<< Modelo compilado >>\n')

print('\n<< Treinando o modelo... >>\n')

history = model.fit([X_train, y_train], [y_train, X_train], batch_size=128, epochs=200, validation_data=[[X_val, y_val], [y_val, X_val]])

model.save('./capsnet_teste.h5')

pd.DataFrame(history.history).to_csv(f'./capsnet_history.csv', index=False)

print('\n<< Treinamento OK >>\n')

print('\n<< Testando o modelo... >>\n')

y_pred, x_recon = model.predict([X_test, y_test], batch_size=100)

c_true = []
c_pred = []

for classe in y_test:
    c_true.append(int(classe[1]))

for classe in y_pred:
    if(classe[0]>classe[1]):
        c_pred.append(0)
    else:
        c_pred.append(1)
        
        
matrix = confusion_matrix(c_true, c_pred)
df_cm = pd.DataFrame(matrix, range(matrix.shape[0]), range(matrix.shape[1]))
sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True, annot_kws={"size": np.max(matrix)})

plt.savefig('./confusion_matrix.svg')

print('\n<< Calculando métricas de validação... >>\n')

lb = LabelBinarizer () 

lb.fit(c_true)

y_test = lb.transform (c_true)
y_pred = lb.transform (c_pred)

auc_score = roc_auc_score(y_test, y_pred,average="macro")
acc = accuracy_score(c_true, c_pred)
kappa = cohen_kappa_score(c_true, c_pred)
recall = recall_score(c_true, c_pred)
prec = precision_score(c_true, c_pred)
f1 = f1_score(c_true, c_pred)

print("Acc = ", acc) 
print("Recall = ", recall)
print("Precision = ", prec)
print("F1-score = ", f1)
print("AUC = ", auc_score)
print("Kappa = ", kappa)

results = {
    'acc': [acc],
    'recall': [recall],
    'precision': [prec],
    'f1-score': [f1],
    'auc': [auc_score],
    'kappa': [kappa]
}

print('\n<< Salvando os resultados... >>\n')

pd.DataFrame(results).to_csv('./results_capsnet.csv', index=False)

print('\n<< Fim :) >>')