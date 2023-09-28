import joblib

model_knn_eucli = joblib.load('knn_model_eucli.pkl')
model_knn_minko = joblib.load('knn_model_minko.pkl')
model_svm_linear = joblib.load('svm_model_linear.pkl')
model_svm_poly3 = joblib.load('svm_model_poly3.pkl')
model_svm_poly4 = joblib.load('svm_model_poly4.pkl')
model_svm_poly5 = joblib.load('svm_model_poly5.pkl')
model_svm_rbf = joblib.load('svm_model_rbf.pkl')

prueba1 = (1,1,17,166,1,135,27,0.957142857, 1.028571429)
prueba2 = (1,1.03125,43.15625,747.34375,17,185,28,0.972972973,1.065099944)
prueba3 = (1,0,49,900,3,125,27,0.955882353,1.023685267)
prueba4 = (1,0,6.90625,161.4882813,12,205,34,0.974025974,1.346504048)
prueba5 = (1,0.313964844,68.74414063,1273.572266,38,170,29,1,1.03596819)
prueba6 = (1,0.5,52.375,859,11,145,25,1,1.145305217)
prueba7 = (0,1,17,166,1,135,27,0.957142857, 1.028571429)
prueba8 = (0,1.03125,43.15625,747.34375,17,185,28,0.972972973,1.065099944)

resultado1 = 1
resultado2 = 1
resultado3 = 0
resultado4 = 0
resultado5 = 2
resultado6 = 1
resultado7 = 1
resultado8 = 2

print("resultado prueba 1: ", resultado1)
print("resultado prueba 2: ", resultado2)
print("resultado prueba 3: ", resultado3)
print("resultado prueba 4: ", resultado4)
print("resultado prueba 5: ", resultado5)
print("resultado prueba 6: ", resultado6)
print("resultado prueba 7: ", resultado7)
print("resultado prueba 8: ", resultado8)

pred1_model_knn_eucli = model_knn_eucli.predict(prueba1)
print("prueba 1", pred1_model_knn_eucli)
pred2_model_knn_eucli = model_knn_eucli.predict(prueba2)
print("prueba 2", pred2_model_knn_eucli)
pred3_model_knn_eucli = model_knn_eucli.predict(prueba3)
print("prueba 3", pred3_model_knn_eucli)
pred4_model_knn_eucli = model_knn_eucli.predict(prueba4)
print("prueba 4", pred4_model_knn_eucli)
pred5_model_knn_eucli = model_knn_eucli.predict(prueba5)
print("prueba 5", pred5_model_knn_eucli)
print()

pred2_model_knn_minko = model_knn_minko.predict(prueba2)
print("prueba 2", pred2_model_knn_eucli)
pred3_model_knn_minko = model_knn_minko.predict(prueba3)
print("prueba 3", pred3_model_knn_minko)
pred4_model_knn_minko = model_knn_minko.predict(prueba4)
print("prueba 4", pred4_model_knn_minko)
pred5_model_knn_minko = model_knn_minko.predict(prueba5)
print("prueba 5", pred5_model_knn_minko)
pred6_model_knn_minko = model_knn_minko.predict(prueba6)
print("prueba 6", pred6_model_knn_minko)
pred7_model_knn_minko = model_knn_minko.predict(prueba7)
print("prueba 7", pred7_model_knn_minko)
print()

pred1_model_svm_linear = model_svm_linear.predict(prueba1)
print("prueba 1", pred1_model_svm_linear)
pred2_model_svm_linear = model_svm_linear.predict(prueba2)
print("prueba 2", pred2_model_svm_linear)
pred3_model_svm_linear = model_svm_linear.predict(prueba3)
print("prueba 3", pred3_model_svm_linear)
pred6_model_svm_linear = model_svm_linear.predict(prueba6)
print("prueba 6", pred6_model_svm_linear)
pred7_model_svm_linear = model_svm_linear.predict(prueba7)
print("prueba 7", pred7_model_svm_linear)
pred8_model_svm_linear = model_svm_linear.predict(prueba8)
print("prueba 8", pred8_model_svm_linear)
print()

pred1_model_svm_poly3 = model_svm_poly3.predict(prueba1)
print("prueba 1", pred1_model_svm_poly3)
pred2_model_svm_poly3 = model_svm_poly3.predict(prueba2)
print("prueba 2", pred2_model_svm_poly3)
pred3_model_svm_poly3 = model_svm_poly3.predict(prueba3)
print("prueba 3", pred3_model_svm_poly3)
"""pred4_model_svm_poly3 = model_svm_poly3.predict(prueba4)
pred5_model_svm_poly3 = model_svm_poly3.predict(prueba5)
pred6_model_svm_poly3 = model_svm_poly3.predict(prueba6)
pred7_model_svm_poly3 = model_svm_poly3.predict(prueba7)
pred8_model_svm_poly3 = model_svm_poly3.predict(prueba8)"""
print()

pred1_model_svm_poly5 = model_svm_poly5.predict(prueba1)
print("prueba 1", pred1_model_svm_poly5)
pred2_model_svm_poly5 = model_svm_poly5.predict(prueba2)
print("prueba 2", pred2_model_svm_poly5)
pred3_model_svm_poly5 = model_svm_poly5.predict(prueba3)
print("prueba 3", pred3_model_svm_poly5)
"""pred4_model_svm_poly5 = model_svm_poly5.predict(prueba4)
pred5_model_svm_poly5 = model_svm_poly5.predict(prueba5)
pred7_model_svm_poly5 = model_svm_poly5.predict(prueba7)"""
print()

pred1_model_svm_rbf = model_svm_rbf.predict(prueba1)
print("prueba 1", pred1_model_svm_rbf)
pred2_model_svm_rbf = model_svm_rbf.predict(prueba2)
print("prueba 2", pred2_model_svm_rbf)
"""pred3_model_svm_rbf = model_svm_rbf.predict(prueba3)
pred4_model_svm_rbf = model_svm_rbf.predict(prueba4)
pred5_model_svm_rbf = model_svm_rbf.predict(prueba5)
pred6_model_svm_rbf = model_svm_rbf.predict(prueba6)
pred7_model_svm_rbf = model_svm_rbf.predict(prueba7)"""
print()


