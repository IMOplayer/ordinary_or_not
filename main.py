from mylib import *

#steps可以是1, 2, 3, 3.1, 4, 4.1, 9, 10, 10.1, 10.2，具體意思請參考代碼
#steps = [1]
#steps = [2, 3.1, 4.1]
#steps = [9]
#steps = [10]
#steps = [10.1]
steps = [10.2]

#是否使用gpu
gpu = False
gpu_nums = "0,1,2"		#使用0,1,2號GPU
gpu_limit = 0.8			#每個GPU的極限鎖定在80%以內

#配置1
'''
src_dir = r"D:\project\pythoncode\cats_and_dogs"		#原始圖片的目錄
dest_dir = r"Z:\Dropbox\project\python_tutorial\keras_cnn\cats_and_dogs"	#目標圖片的目錄
extension = "jpg"	#圖像文件的擴展名
reselect = True		#是否重新選取圖片

train_n = 2000		#隨機選取多少個訓練圖片
val_n = 1000		#隨機選取多少個驗證圖片
test_n = 1000		#隨機選取多少個測試圖片
pred_n = 10			#預測的目錄中有多少張圖片

class_mode = "binary"			#二分類
#class_mode = "categorical"	#多分類
class_n = 2						#分類的數目

train_dir = r"./cats_and_dogs/train"
val_dir = r"./cats_and_dogs/val"
test_dir = r"./cats_and_dogs/test"
pred_dir = r"./cats_and_dogs/pred"

model_file = "cats_and_dogs"	#訓練後的模型文件
width = 150			#放縮後圖片的大小
batch_size = 32		#批次的圖片數量
epochs = 30			#訓練多少次
patience = 5		#多少次沒有轉好才停止
'''

#配置2
src_dir = r"D:\project\pythoncode\animals10\raw-img"		#原始圖片的目錄
dest_dir = r"Z:\Dropbox\project\python_tutorial\keras_cnn\animals10"	#目標圖片的目錄
extension = "jpg"	#圖像文件的擴展名
reselect = False	#是否重新選取圖片

train_n = 634		#隨機選取多少個訓練圖片
val_n = 264			#隨機選取多少個驗證圖片
test_n = 106		#隨機選取多少個測試圖片
pred_n = 2			#預測的目錄中有多少張圖片

#class_mode = "binary"			#二分類
class_mode = "categorical"		#多分類
class_n = 3		#分類的數目

train_dir = r"./headshot/train"
val_dir = r"./headshot/val"
test_dir = r"./headshot/test"
pred_dir = r"C:\Users\USER\Desktop\ordinary_or_not\user_headshot"

model_file = "images2"
width = 400		#放縮後圖片的大小
batch_size = 16	#批次的圖片數量
epochs = 15		#訓練多少次
patience = 3		#多少次沒有轉好才停止


#--------------------------------------------------------
#	GPU處理
#--------------------------------------------------------
if gpu:
	import tensorflow as tf
	import keras.backend.tensorflow_backend as KTF

	os.environ["CUDA_VISIBLE_DEVICES"] = gpu_nums    #使用0,1,2號GPU
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = gpu_limit #每個GPU的極限鎖定在80%以內
	session = tf.compat.v1.Session(config=config)

	#設定GPU，tensorflow版本處理
	if tf.__version__ >= "2.0.0":
		tf.compat.v1.keras.backend.set_session(session)	#tensorflow 2
	else:
		KTF.set_session(session)        				#tensorflow 1

#--------------------------------------------------------
#	轉移圖片
#--------------------------------------------------------
os.chdir(os.path.dirname(__file__) or './')
def cnn(file, steps):
	if steps is None:
		steps = steps
	if 1 in steps:
		import sys
		set_images(src_dir, dest_dir, train_n//class_n, val_n//class_n, test_n//class_n, reselect, extension)
		sys.exit()

	#--------------------------------------------------------
	#	建立模型
	#--------------------------------------------------------
	#from tensorflow import keras
	from tensorflow.keras import models, layers, optimizers
	if 2 in steps:
		#TODO，以下模型可全部修改
		model = models.Sequential()
		model.add(layers.Conv2D(64, (3, 3), input_shape=(width, width, 3)))
		model.add(layers.Activation("relu"))
		model.add(layers.MaxPooling2D((2, 2)))

		model.add(layers.Conv2D(32, (3, 3)))
		model.add(layers.Activation("relu"))
		model.add(layers.MaxPooling2D((2, 2)))

		model.add(layers.Conv2D(16, (3, 3)))
		model.add(layers.Activation("relu"))
		model.add(layers.MaxPooling2D((2, 2)))

		model.add(layers.Flatten())
		model.add(layers.Dense(4096))
		model.add(layers.Activation("relu"))
		model.add(layers.Dropout(0.2))

	#二分類
	if 2 in steps and class_mode=="binary":
		model.add(layers.Dense(1))		#二分類時為1
		model.add(layers.Activation("sigmoid"))
		model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
					  loss="binary_crossentropy",
					  metrics=["acc"])
		model.summary()

	#多分類
	if 2 in steps and class_mode=="categorical":
		model.add(layers.Dense(class_n))
		model.add(layers.Activation("softmax"))
		model.compile(loss="categorical_crossentropy",
					  optimizer=optimizers.RMSprop(lr=1e-4),
					  metrics=["acc"])
		model.summary()


	#--------------------------------------------------------
	#	製作圖像生成器
	#--------------------------------------------------------
	from keras_preprocessing.image import ImageDataGenerator
	import pickle

	#不擴增圖像
	if 3 in steps:
		train_datagen = ImageDataGenerator(rescale=1./255)
		val_datagen = ImageDataGenerator(rescale=1./255)

	#TODO，使用圖像擴增法
	if 3.1 in steps:
		train_datagen = ImageDataGenerator(
							rescale				= 1./255,
							rotation_range 		= 40,
							width_shift_range	= 0.2,
							height_shift_range	= 0.2,
							zoom_range			= 0.2,
							horizontal_flip		= True,		#路標就不能左右反轉
							fill_mode			= "nearest"
						)
		val_datagen = ImageDataGenerator(rescale=1./255)

	#製作生成器
	if 3 in steps or 3.1 in steps:
		train_generator = train_datagen.flow_from_directory(
			train_dir,
			target_size = (width, width),
			batch_size = batch_size,
			class_mode = class_mode,
		)
		val_generator = val_datagen.flow_from_directory(
			val_dir,
			target_size = (width, width),
			batch_size = batch_size,
			class_mode = class_mode,
		)
		train_n = len(train_generator.filenames)	#有需要時才使用
		val_n = len(val_generator.filenames)

		#儲存標記分類
		with open(model_file + "_labels.pickle", "wb") as op:
			pickle.dump(train_generator.class_indices, op)

		print("** %d images will be trained" %(train_n))
		print("** %d images will be validated" %(val_n))


	#--------------------------------------------------------
	#	訓練模型
	#--------------------------------------------------------
	#訓練epochs次
	if 4 in steps:
		#訓練
		history = model.fit_generator(        #開始訓練，新版是fit
			train_generator,
			steps_per_epoch = train_n // batch_size,	#梯度下降次數，和訓練圖像數目有關
			epochs = epochs,
			validation_data = val_generator,
			validation_steps = val_n // batch_size,
			#workers = 3,                  #TODO，開多少線程，不可以和use_multiprocessing同時打開
			#use_multiprocessing = True    #TODO，是否並行處理
		)

	#最多訓練epochs次，若最好結果在patience次後沒改進，就自動停止，能省時間
	if 4.1 in steps:
		from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

		#停止訓練的時機
		es = EarlyStopping(monitor='val_loss', mode='min', patience=patience, verbose=1)
		mc = ModelCheckpoint(model_file+".h5", monitor='val_loss', mode='min', save_best_only=True, verbose=1)

		#訓練
		history = model.fit_generator(        #開始訓練，新版是fit
			train_generator,
			steps_per_epoch = train_n // batch_size,
			epochs = epochs,
			validation_data = val_generator,
			validation_steps = val_n // batch_size,
			callbacks = [es, mc],
			#workers = 3,                  #TODO，開多少線程，不可以和use_multiprocessing同時打開
			#use_multiprocessing = True    #TODO，是否並行處理
		)
		print("[*] %s saved." %(model_file+".h5"))

	#儲存模型，記錄訓練的過程
	if 4 in steps or 4.1 in steps:
		model.save(model_file + ".h5")
		with open(model_file+".pickle", "wb") as op:	#記錄
			#pickle.dump(history, op)
			pickle.dump(history.history, op)
		print("[*] %s saved." %(model_file+".pickle"))


	#====================================================
	#	查看訓練過程
	#====================================================
	from PIL import Image
	import numpy as np
	if 9 in steps:
		import matplotlib.pyplot as plt

		#讀取訓練過程
		with open(model_file+".pickle", "rb") as op:
			history = pickle.load(op)

		#提取數據
		acc = history["acc"]	#舊版是history.history
		val_acc = history["val_acc"]
		loss = history["loss"]
		val_loss = history["val_loss"]
		epochs_arr = range(1, len(acc)+1)

		#畫圖
		fig, ax = plt.subplots(1, 2)
		plt.tight_layout(pad=2, w_pad=2)
		ax[0].set_title("Training and Validation Accuracy")
		ax[0].plot(epochs_arr, acc, "b", linewidth=1, label="Training acc")
		ax[0].plot(epochs_arr, val_acc, "r", linewidth=1, label="Validation acc")
		ax[0].legend()
		ax[1].set_title("Training and Validation Loss")
		ax[1].plot(epochs_arr, loss, "b", linewidth=1, label="Training loss")
		ax[1].plot(epochs_arr, val_loss, "r", linewidth=1, label="Validation loss")
		ax[1].legend()
		plt.show()


	#====================================================
	#	測試
	#====================================================
	if 10 in steps or 10.1 in steps or 10.2 in steps:
		if 10.1 in steps:
			test_dir = pred_dir

		#讀取分類
		with open(model_file + "_labels.pickle", "rb") as op:
			labels = pickle.load(op)


	if 10 in steps or 10.1 in steps:
		#圖像生成器
		test_datagen = ImageDataGenerator(rescale=1./255)
		test_generator = test_datagen.flow_from_directory(
			directory = test_dir,
			target_size = (width, width),
			batch_size = 1,
			class_mode = class_mode,
			shuffle = False,
		)
		#載入模型
		model = models.load_model(model_file + ".h5")

	'''測試loss和acc'''
	if 10 in steps:
		#預測
		test_loss, test_acc = model.evaluate_generator(	#新版是evaluate
								test_generator,
								steps = test_n,
								verbose = 1)
		print("[*] test loss:", test_loss)
		print("[*] test acc:", test_acc)

	'''預測目錄內所有圖像'''
	if 10.1 in steps:
		#預測
		pred = model.predict_generator(test_generator, steps=pred_n)

		#顯示結果
		files = test_generator.filenames
		labels = dict((v,k) for (k,v) in labels.items())	#調換鍵值
		if class_mode=="binary":
			indices = (pred.flatten()>0.5).astype(np.int)
		elif class_mode=="categorical":
			indices = np.argmax(pred, axis=1)
		print("** {} images are predicted: ".format(len(files)))
		for i in range(0, len(files)):
			print("{}\t{}".format(files[i], labels[indices[i]]))

	'''TODO，修改後可用於預測單幅圖像'''
	if 10.2 in steps:
		from tensorflow.keras.preprocessing import image #舊版
		#import tensorflow.keras.utils as image

		#載入模型
		model = models.load_model(model_file + ".h5")
		test_datagen = ImageDataGenerator(rescale=1./255)

		#生成陣列
		paths = glob.glob(os.path.join(r"user_headshot/"+file))
		print("** {} images are predicted: ".format(len(paths)))

		#逐幅圖像去判斷
		labels = dict((v,k) for (k,v) in labels.items())	#調換鍵值
		for path in paths:
			#推薦到圖像矩陣
			img = image.load_img(path, target_size=(width, width))
			img = image.img_to_array(img)
			img = np.expand_dims(img, axis=0)

			#生成器
			test_generator = test_datagen.flow(img)
			test_generator.reset()		#不reset好像也不會影響結果

			#預測
			pred = model.predict(test_generator, steps=1)	#新版用predict

			#顯示結果
			if class_mode=="binary":
				k = (pred.flatten()>0.5).astype(np.int)[0]
			elif class_mode=="categorical":
				k = np.argmax(pred)
			print("{}\t{}".format(path, labels[k]))
			return labels[k]

