# Elyazılarından Yazar Tanımlama

Elyazılarından yazar tanımlama işlemi belgelerin adli analizinde önemli bir konu olmuştur.
Yazar kimliğinin ve yazarın ele geçirilmesindeki zorluklar;

bir kişinin yazı stilini, 

yazarın fiziksel durumunu, 

çoklu görev ve gürültü gibi dikkatleri değiştiren farklı kalemlerin kullanımını 
yazma stilinin yaşla birlikte değişmesini de içerir.
Yazar kimliği, adli tıp alanındaki yazılarda olduğu gibi, yazıların eskilerle karşılaştırılması  gerektiğinde, farklı harfler arasındaki bağlantıların kurulabilmesi için de kullanılabilir. 
Yazar tanıma üzerinde birçok yöntem geliştirilmiş  ve test edilmiştir. 


DeepWriter: Metin Bağımsız Yazar Kimlik Doğrulama için Çok Akışlı Derin CNN


Off-line yazar tanımlaması için yöntemler iki gruba ayrılabilir: metne bağımlı ve metinden bağımsız. metin bağımlı yöntemler;
Metin bağımlı: Tüm yazarların aynı metni yazması gerekmekte.
Metin bağımsız: Hiçbir kısıtlama yok. Eğitim ve test için yazara ait herhangi elyazısı olması yeterli.
Bununla birlikte, metne bağımlı bir metinle karşılaştırıldığında, metinden bağımsız yazar tanımlaması, görüntünün devasa kategoriler arası çeşitlilik sergileyen keyfi metinlerle uğraşması gerekir ve bu nedenle de daha zorlayıcıdır. 
Bu zorlu soruna hitap etmek için, bu yazı, off-line metin bağımsız yazar tanımlaması için etkili sunumları öğrenmek için güçlü bir model olarak Deep CNN (Convolutional Neural Network) kullanmaktadır. 


**Deep CNN yapısı**, en yeni bilgisayar görüntü problemlerinde, görüntü sınıflandırması, nesne algılama ve  yüz tanıma  dahil olmak üzere büyük bir etki göstermiştir. 

El yazısı tanıma  vb. yazarlar arası tanımlama özelliklerini ayıklamak için çok kanallı bir CNN olan DeepWriter'i öneriyoruz. 


Giriş görüntüsünü sabit boyutuna yeniden boyutlandırma, el yazısı şeklini bozar ve ciddi bilgi kaybına yol açar. Bu nedenle bu sorunu çözmek için bir parça tarama stratejisi kullanıyoruz.
Veri seti olarak verilen elyazıları öncelikle, satırlara ayrılır. 
Ayrılan satırlardan 113x113 boyutlarında parçalar çıkartılarak eğitim ve test için rastgele olarak bölünür.
Ayrılan 113x113 boyutundaki parçalar resize() fonksiyonu ile program içerisinde 56x56 olarak yeniden boyutlandırılır. 
Buradaki amaç hespalamayı kolaylaştırmaktır.

![](https://github.com/ozdenurucar/HandwriterIdentification/blob/master/Images/patchs.png)


*Bu paketleri projeye aktarın.

[source,python]
```
from __future__ import division
import numpy as np
import os
import glob

from random import *
from PIL import Image
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
%matplotlib inline

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMSprop
```
Bunlar, veri kümesindeki her bir sütunda dosya adlarının değiştirilmesinden hızlı erişim için kullanılan formlardır. Form ve yazar eşleştirme ile bir dictionary oluşturalım.

```
d = {}
from subprocess import check_output
with open('drive/Colab_Kullanim/WriterIdentification/forms_for_parsing.txt') as f:
    for line in f:
        key = line.split(' ')[0]
        writer = line.split(' ')[1]
        d[key] = writer
print(len(d.keys()))
```
Tüm dosya adları listesi ve hedef yazar adları listesi oluşturulur.

```
tmp = []
target_list = []

path_to_files = os.path.join('drive/Colab_Kullanim/WriterIdentification/data_subset', '*')
for filename in sorted(glob.glob(path_to_files)):
#     print(filename)
    tmp.append(filename)
    image_name = filename.split('/')[-1]
    file, ext = os.path.splitext(image_name)
    parts = file.split('-')
    form = parts[0] + '-' + parts[1]
    for key in d:
        if key == form:
            target_list.append(str(d[form]))

img_files = np.asarray(tmp)
img_targets = np.asarray(target_list)
print(img_files.shape)
print(img_targets.shape)
```

Görüntü verilerini görselleştirelim.

```
for filename in img_files[:3]:
    img=mpimg.imread(filename)
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap ='gray')
```

Kategorik veri olmadığını görmek güzel. Böylece normalizasyon etiket kodlayıcı kullanılarak yapılır.

```
train_files, rem_files, train_targets, rem_targets = train_test_split(
        img_files, encoded_Y, train_size=0.66, random_state=52, shuffle= True)

validation_files, test_files, validation_targets, test_targets = train_test_split(
        rem_files, rem_targets, train_size=0.5, random_state=22, shuffle=True)

print(train_files.shape, validation_files.shape, test_files.shape)
print(train_targets.shape, validation_targets.shape, test_targets.shape)
```

**Modele giriş**

Daha önce de söylediğimiz gibi, her biri 113x133 boyutunda veri yamalarını alıyoruz. Bu amaç için bir jeneratör fonksiyonu uygulanmaktadır.

```
batch_size = 8 #16
num_classes = 50


# Start with train generator shared in the class and add image augmentations
def generate_data(samples, target_files,  batch_size=batch_size, factor = 0.1 ):
    num_samples = len(samples)
    from sklearn.utils import shuffle
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_targets = target_files[offset:offset+batch_size]

            images = []
            targets = []
            for i in range(len(batch_samples)):
                batch_sample = batch_samples[i]
                batch_target = batch_targets[i]
                im = Image.open(batch_sample)
                cur_width = im.size[0]
                cur_height = im.size[1]

                # print(cur_width, cur_height)
                height_fac = 113 / cur_height

                new_width = int(cur_width * height_fac)
                size = new_width, 113

                imresize = im.resize((size), Image.ANTIALIAS)  # Resize so height = 113 while keeping aspect ratio
                now_width = imresize.size[0]
                now_height = imresize.size[1]
                # Generate crops of size 113x113 from this resized image and keep random 10% of crops

                avail_x_points = list(range(0, now_width - 113 ))# total x start points are from 0 to width -113

                # Pick random x%
                pick_num = int(len(avail_x_points)*factor)

                # Now pick
                random_startx = sample(avail_x_points,  pick_num)

                for start in random_startx:
                    imcrop = imresize.crop((start, 0, start+113, 113))
                    images.append(np.asarray(imcrop))
                    targets.append(batch_target)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(targets)

            #reshape X_train for feeding in later
            X_train = X_train.reshape(X_train.shape[0], 113, 113, 1)
            #convert to float and normalize
            X_train = X_train.astype('float32')
            X_train /= 255

            #One hot encode y
            y_train = to_categorical(y_train, num_classes)
            yield shuffle(X_train, y_train)
```
