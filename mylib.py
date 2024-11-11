import os, glob, shutil
import random

def set_images(src, dest, trainnum=1000, valnum=500, testnum=500, reset=False, extension="jpg"):
	#創建目錄
	if reset and os.path.isdir(dest):
		shutil.rmtree(dest)
	
	dest_train = os.path.join(dest, "train") 
	dest_val = os.path.join(dest, "val")
	dest_test = os.path.join(dest, "test")
	if not os.path.isdir(dest):
		os.mkdir(dest)
	if os.path.isdir(dest_train) and os.path.isdir(dest_val) and os.path.isdir(dest_test):
		return	#目錄已存在就不需要重新新生圖片
		
	if not os.path.isdir(dest_train):
		os.mkdir(dest_train)
	if not os.path.isdir(dest_val):
		os.mkdir(dest_val)
	if not os.path.isdir(dest_test):
		os.mkdir(dest_test)		
		
	#遍歷目錄
	for folder in os.listdir(src):
		src_root = os.path.join(src, folder)
		imgs_n = len(os.listdir(src_root))
		
		#隨機數字，用於生成隨機圖像
		nums = random.sample(range(0,imgs_n), trainnum+valnum+testnum)
		nums_train = nums[:trainnum]
		nums_val = nums[trainnum:trainnum+valnum]
		nums_test = nums[trainnum+valnum:trainnum+valnum+testnum]
		
		#複製圖像
		os.mkdir(os.path.join(dest_train, folder))
		os.mkdir(os.path.join(dest_val, folder))
		os.mkdir(os.path.join(dest_test, folder))
		for i in nums_train:
			nm = "{}.{}.{}".format(folder, i, extension)
			shutil.copyfile(os.path.join(src,folder,nm), os.path.join(dest_train,folder,nm))
		for i in nums_val:
			nm = "{}.{}.{}".format(folder, i, extension)
			shutil.copyfile(os.path.join(src,folder,nm), os.path.join(dest_val,folder,nm))
		for i in nums_test:
			nm = "{}.{}.{}".format(folder, i, extension)
			shutil.copyfile(os.path.join(src,folder,nm), os.path.join(dest_test,folder,nm))

	print("[*] images copied! \n")
	
	