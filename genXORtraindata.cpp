// referring this page [https://medium.com/@shiyan/caffe-c-helloworld-example-with-memorydata-input-20c692a82a22#.cexyka70v] for the example provided below.
#include<iostream>
#include<stdlib.h>

#include<caffe/caffe.hpp>
#include<caffe/layers/memory_data_layer.hpp>

using namespace caffe;
using namespace std;

int main(int argc, char* argv[]){

	if(argc<3){
		cout<<"Requires more arguments.\n"
		<<"Correct format: ./genXORtraindata solver.prototxt inputDataLayerName\n";
		return(-1);
	}

	// generate 400 sets of training data. Each training data has the batch size of 64
	float *data = new float[64*1*1*2*400];
	float *label = new float[64*1*1*1*400];
	
	for(int i=0; i<64*1*1*1*400; i++){
		int a = rand() % 2;
		int b = rand() % 2;
		int c = a ^ b;
		data[i*2 + 0] = a;
		data[i*2 + 1] = b;
		label[i] = c;
	}
	
	// caffe is using google logging (aka 'glog') as its logging module, and hence this module must be initialized once when running caffe. Therefore the following line 
	::google::InitGoogleLogging(argv[0]);
	
	// create a solver parameter object and load solver.prototxt into it
	SolverParameter solver_param;
	ReadSolverParamsFromTextFileOrDie(argv[1], &solver_param);
	
	// create the solver out of the solver parameter
	shared_ptr<Solver<float> > // uses namespace std
	solver(SolverRegistry<float>::CreateSolver(solver_param));
	
	// obtain the input MemoryData layer from the solver’s neural network and feed in the training data
	MemoryDataLayer<float> *dataLayer_trainnet = 
	(MemoryDataLayer<float> *)
	(solver->net()->layer_by_name(argv[2]).get());
	dataLayer_trainnet->Reset(data, label, 25600);
	// The reset function of MemoryData allows us to provide pointers to the memory of data and labels. Again, the size of each label can only be 1, whereas the size of data is specified in the model.prototxt file. 25600 is the count of training data. It has to be a multiply of 64, the batch size. 25600 is 400 * 64. Basically we generated 400 training data with batch size of 64.
	
	// train the network
	cout<<"Please wait, training the network...";
	solver->Solve();
	cout<<"done!\n";	

	return(0);
}
