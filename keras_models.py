import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import inception_v3, vgg16, resnet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
from PIL import Image
import urllib2

URL = ['http://static.flickr.com/101/315134625_8d1110047e.jpg',
       'http://farm1.static.flickr.com/180/376380977_62b9c50607.jpg',
       'http://farm1.static.flickr.com/50/137829272_40de8efe97.jpg',
       'http://farm1.static.flickr.com/34/121568295_797f01c582.jpg',
       'http://farm3.static.flickr.com/2222/2066536533_01c90e0cfa.jpg',
       'http://static.flickr.com/1048/917051870_b7b88e9bb6.jpg',
       'http://farm2.static.flickr.com/1038/784726007_21a0dc6aac.jpg',
       'http://farm1.static.flickr.com/56/164295610_a4f44fcf2e.jpg',
       'http://static.flickr.com/3260/2713784719_0698b5ec43.jpg',
       'http://farm1.static.flickr.com/175/442811443_67528f7e1e.jpg']

cont=0
for url in URL:
  
    req = urllib2.Request(url)
    response = urllib2.urlopen(req)
    f=open('test'+str(cont)+'.jpg', 'w');
    f.write(response.read())
    cont=cont+1


#precisa carregar os modelos com os pesos correspondentes da base de dados imagenet. Isso eh feito para cada modelos carregado

modelo_vgg16 = vgg16.VGG16(weights='imagenet')
modelo_inception = inception_v3.InceptionV3(weights='imagenet')
modelo_resnet = resnet50.ResNet50(weights='imagenet')

for i in range(0,cont-1,1):
	caminho_arquivo = 'test'+str(i)+'.jpg'

	# carregar uma imagem
	img = load_img(caminho_arquivo,target_size=(224, 224))

		 
	# converter a imagem em formato PIL para um tipo numpy

	numpy_img = img_to_array(img)

	 
	# Converte a imagem para um formato batch, expand_dims vai adicionar uma dimensao extra para o eixo 0
	#A entrada da rede tera um formato (tamanho de batch, altura, largura, canais)
	img_batch = np.expand_dims(numpy_img, axis=0)

	# prepara a imagem para os modelos
	imagem_processada_vgg = vgg16.preprocess_input(img_batch.copy())
	imagem_processada_inception = inception_v3.preprocess_input(img_batch.copy())
	imagem_processada_resnet = resnet50.preprocess_input(img_batch.copy())

	# faz a predicao para cada classe
	predicoes_vgg = modelo_vgg16.predict(imagem_processada_vgg)
	predicoes_inception = modelo_inception.predict(imagem_processada_inception)
	predicoes_resnet = modelo_resnet.predict(imagem_processada_resnet)
	# converte as probabilidades para classes rotuladas (diz a procentagem para o que e leao,passaro,gato e etc)
	# pegando apenas as cinco melhores predicoes
	label_vgg = decode_predictions(predicoes_vgg)
	label_inception = decode_predictions(predicoes_inception)
	label_resnet = decode_predictions(predicoes_resnet)
        print("Imagem de teste "+str(i))
        print("============================================")
	print('modelo vgg')
	print('Classe: ',label_vgg[0][0][1])
	print('Porcentagem: ',label_vgg[0][0][2])
	print("============================================")
	print('modelo inception')
	print('Classe: ',label_inception[0][0][1])
	print('Porcentagem: ',label_inception[0][0][2])
	print("============================================")
	print('modelo resnet')
	print('Classe: ',label_resnet[0][0][1])
	print('Porcentagem: ',label_resnet[0][0][2])
	print("============================================")
        print("*********************************************")
        print("*********************************************")


