#Load the relevant packages,
library(keras)
install_keras()


suppressMessages(library(pROC))
suppressMessages(library(caret))
library(tensorflow)

set.seed(123)

#The train generator uses augmentation,
train_datagen = image_data_generator(
  rescale = 1/255,
  rotation_range = 5,
  width_shift_range = 0.1,
  height_shift_range = 0.05,
  shear_range = 0.1,
  zoom_range = 0.15,
  horizontal_flip = TRUE,
  vertical_flip = FALSE,
  fill_mode = "reflect"
)



validation_datagen <- image_data_generator(rescale = 1/255)  
test_datagen <- image_data_generator(rescale = 1/255)  

training_batch_size = 32
validation_batch_size = 32

train_path="C:/Users/admin/Pictures/scinokine/visiting_card/corona_images/final_data/train/"


train_generator <- flow_images_from_directory(
  train_path,                            # Target directory  
  train_datagen,                        # Data generator
  classes = c('NORMAL', 'COVID'),
  target_size = c(224, 224),            # Resizes all images
  batch_size = training_batch_size,
  class_mode = "categorical",
  shuffle = T,
  seed = 123
)

val_path="C:/Users/admin/Pictures/scinokine/visiting_card/corona_images/final_data/val/"


validation_generator <- flow_images_from_directory(
  val_path,
  classes = c('NORMAL', 'COVID'),
  validation_datagen,
  target_size = c(224, 224),
  batch_size = validation_batch_size,
  class_mode = "categorical",
  shuffle = T,
  seed = 123
)

test_path="C:/Users/admin/Pictures/scinokine/visiting_card/corona_images/final_data/test/"

test_generator <- flow_images_from_directory(
  test_path,
  classes = c('NORMAL', 'COVID'),
  test_datagen,
  target_size = c(224, 224),
  batch_size = 1,
  class_mode = "categorical",
  shuffle = FALSE
)



conv_base <- application_inception_resnet_v2(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(224, 224, 3)
)




#Create a new top,
model <- keras_model_sequential() %>% 
  conv_base %>% 
  layer_global_average_pooling_2d(trainable = T) %>%
  layer_dropout(rate = 0.2, trainable = T) %>%
  layer_dense(units = 224, activation = "relu", trainable = T) %>% 
  layer_dense(units = 2, activation = "softmax", trainable = T)

#Don't train the base,
freeze_weights(conv_base)

set.seed(123)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-6),
  metrics = c("accuracy")
)


summary(model)

training_step_size = ceiling(length(list.files(train_path, recursive = T)) / training_batch_size)
validation_step_size = ceiling(length(list.files(val_path, recursive = T)) / validation_batch_size)


#Not sure if this line is stupid or not. I'm trying to address the class weights imbalance,
covid_adjustment = length(list.files(paste(train_path, '/NORMAL/', sep = ""), recursive = T)) / length(list.files(paste(train_path, '/COVID/', sep = ""), recursive = T))



set.seed(123)

history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 25,
  class_weight = list("0"=1,"1"=covid_adjustment),
  epochs = 10,
  validation_data = validation_generator,
  validation_steps = ceiling(length(list.files(val_path, recursive = T)) / validation_batch_size)
)


plot(history)


preds = predict_generator(model,
                          test_generator,
                          steps = length(list.files(test_path, recursive = T)))


predictions = data.frame(test_generator$filenames)
predictions$prob_covid = preds[,2]
colnames(predictions) = c('Filename', 'Prob_covid')

head(predictions, 10)

predictions$Class_predicted = 'Normal'
predictions$Class_predicted[predictions$Prob_covid >= 0.5] = 'Covid'
predictions$Class_actual = 'Normal'
predictions$Class_actual[grep("COVID", predictions$Filename)] = 'Covid'


predictions$Class_actual = as.factor(predictions$Class_actual)
predictions$Class_predicted = as.factor(predictions$Class_predicted)


roc = roc(response = predictions$Class_actual, 
          predictor = as.vector(predictions$Prob_covid), 
          ci=T,
          levels = c('Normal', 'Covid'))
threshold = coords(roc, x = 'best', best.method='youden')
threshold



boxplot(predictions$Prob_covid ~ predictions$Class_actual,
        main = 'Probabilities of Normal vs Covid Patients',
        ylab = 'Probability',
        col = 'light blue')
abline(h=threshold[1], col = 'red', lwd = 3)

predictions$Class_predicted = 'Normal'
predictions$Class_predicted[predictions$Prob_covid >= threshold[1]] = 'Covid'

str(predictions$Class_actual)
str(predictions$Class_predicted)

library(caret)
cm = confusionMatrix(predictions$Class_actual,predictions$Class_predicted)
cm


paste("AUC =", round(roc$auc, 3))
plot(roc)
ci_sens = ci.se(roc, specificities = seq(from=0, to=1, by=0.01), boot.n = 2000)
plot(ci_sens, type="shape", col="#00860022", no.roc=TRUE)
#abline(h=cm$byClass['Sensitivity'], col = 'red', lwd = 2)
#abline(v=cm$byClass['Specificity'], col = 'red', lwd = 2)


thresholds = coords(roc, x = "local maximas")
thresholds = as.data.frame(t(thresholds))
thresholds[(thresholds$sensitivity > 0.85 & thresholds$sensitivity < 0.9),]

saveRDS(model, "CovidDiagnosisModel")
