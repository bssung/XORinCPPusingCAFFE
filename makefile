train:
	g++ `pkg-config --cflags --libs opencv gflags libglog` genXORtraindata.cpp -o genXORtraindata -DCPU_ONLY -lboost_system -L/nfs/engine/milindp/localInstalls/caffe/build/lib -lcaffe
	
test:
	g++ `pkg-config --cflags --libs opencv gflags libglog` genXORtestdata.cpp -o genXORtestdata -DCPU_ONLY -lboost_system -L/nfs/engine/milindp/localInstalls/caffe/build/lib -lcaffe
	
cleanmodels:
	rm *.caffemodel
	
cleanstates:
	rm *.solverstate
