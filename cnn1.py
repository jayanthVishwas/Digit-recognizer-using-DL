from keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

import matplotlib.pyplot as plt


# print(x_train.shape)
# print(x_test.shape)

# single_image=x_train[0]

# # plt.imshow(single_image)
# # plt.show()

# # print(single_image)

# # print(single_image.shape)

# #pre processing data

# print(y_train)
# print(y_test)

# from keras.utils.np_utils import to_categorical

# print(y_train.shape)

# y_cat_train=to_categorical(y_train,10)

# print(y_cat_train.shape)

# y_cat_test=to_categorical(y_test,10)

# print(y_cat_test.shape)

# #normalize data

# print(single_image.max())

# print(single_image.min())


# x_train=x_train/255
# x_test=x_test/255

# scaled_single=x_train[0]
# # print(scaled_single.max())

# # plt.imshow(scaled_single)
# # plt.show()

# #reshape the data
# bk_x_test=x_test
# x_train=x_train.reshape(60000,28,28,1)
# x_test=x_test.reshape(10000,28,28,1)


# print(x_train.shape)
# print(x_test.shape)

# #train the model

# from keras.models import Sequential
# from keras.layers import Dense,Conv2D, MaxPool2D,Flatten

# model=Sequential()

# model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(28,28,1),activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Flatten())
# model.add(Dense(128,activation='relu'))
# model.add(Dense(10,activation='softmax'))

# model.compile(loss='categorical_crossentropy',
# optimizer='rmsprop',
# metrics=['accuracy']
# )

# print(model.summary())

# model.fit(x_train,y_cat_train,epochs=2)

# model.evaluate(x_test,y_cat_test)

# from sklearn.metrics import classification_report

# predictions=model.predict_classes(x_test)

# print(y_test[0])
# plt.imshow(bk_x_test[0])
# plt.show()


# import cv2
# import numpy as np
# from keras.preprocessing import image

# drawing=False

# point_x,point_y=None,None

# def drawing_screen(event,x,y,flags,param):
    # global point_x,point_y,drawing
    
    # if event==cv2.EVENT_LBUTTONDOWN:
        # drawing = True
        # point_x,point_y=x,y 
    
    # elif event==cv2.EVENT_MOUSEMOVE:
        # if drawing==True:
            # cv2.line(img,(point_x,point_y),(x,y),color=(255,255,255),thickness=3)
            # point_x,point_y=x,y
    
    # elif event==cv2.EVENT_LBUTTONUP:
        # drawing=False
        # cv2.line(img,(point_x,point_y),(x,y),color=(255,255,255),thickness=3)


# img=np.zeros((512,512,3),np.uint8)
# cv2.namedWindow('test draw')
# cv2.setMouseCallback('test draw',drawing_screen)
        

# while(1):
    # cv2.imshow('test draw',img)
    # if cv2.waitKey(2) & 0xFF == ord('s'):
        # cv2.imwrite('C:/python/paint.png',img)
        # testing = image.load_img('C:/python/paint.png',target_size=(28,28))
        # testing=image.img_to_array(testing)
        # testing=cv2.cvtColor(testing,cv2.COLOR_BGR2GRAY)
        # k=model.predict_classes(testing.reshape(1,28,28,1))
        # print(k[0])
    # elif cv2.waitKey(2) & 0xFF==ord('n'):
        # img = np.zeros((512,512,3),np.uint8)
    # elif cv2.waitKey(2) &0xFF==27:
        # break
# cv2.destroyAllWindows()





#testing = 'C:/python/paint.png'

#predictions=model.predict_classes(x_test)


plt.imshow("paint1.png")
plt.show()










