// moeo general include
#define M_PI 3.14159265358979323846
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <assert.h>
#include <stdlib.h>
#include <vector>
#include <stdexcept>

#include <iostream>
#include <fstream>
#include <moeo>

#include <es/eoRealOp.h> 
#include <es/eoRealInitBounded.h>

// how to initialize the population
#include <do/make_pop.h>
// the stopping criterion
#include <do/make_continue_moeo.h>
// outputs (stats, population dumps, ...)
#include <do/make_checkpoint_moeo.h>
// evolution engine (selection and replacement)
#include <do/make_ea_moeo.h>
// simple call to the algo
#include <do/make_run.h>

// crossover and mutation operators
#include <SBXCrossover.h>
#include <PolynomialMutation.h>

using namespace std;

PyObject *pName, *pModule, *pFunc;
const char *pythonScript = "loadAndRun"; //Name of the python script
const char *pythonModule = "queryModel"; //Module in the python script
long int num_FE_used = 0;
long int max_FE = 0;
float penalty_weight = 0.1;

// PyObject -> Vector
vector<double> listTupleToVector_Float(PyObject* incoming) {
	vector<double> data;
	if (PyTuple_Check(incoming)) {
		for(Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) {
			PyObject *value = PyTuple_GetItem(incoming, i);
			data.push_back( PyFloat_AsDouble(value) );
		}
	} else {
		if (PyList_Check(incoming)) {
			for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
				PyObject *value = PyList_GetItem(incoming, i);
				data.push_back( PyFloat_AsDouble(value) );
			}
		} else {
			throw logic_error("Passed PyObject pointer was not a list or tuple!");
		}
	}
	return data;
}

// Initialize and test the python program with the embedded objective function
bool initPythonStuff(bool ready) {
     PyObject *pArgs, *pValue;
     bool pythonReady = false;
     std::vector<double> valuesForResult;
     std::vector<double> valuesForTest;
     valuesForTest.push_back(0);
     valuesForTest.push_back(0.42);
     valuesForTest.push_back(2);
     valuesForTest.push_back(0);
     valuesForTest.push_back(1);
     valuesForTest.push_back(0);
     valuesForTest.push_back(790);
     valuesForTest.push_back(5);
     valuesForTest.push_back(5);

     if (!ready){
          Py_Initialize();
          PyRun_SimpleString("import sys");
          PyRun_SimpleString("import sys\nsys.path.append(\"python\")\n" ); //add paths where python files are
          PyRun_SimpleString("sys.path.append(\".\")");
          pName = PyUnicode_DecodeFSDefault(pythonScript);
          pModule = PyImport_Import(pName);
     }
     
     if (pModule != NULL) {
          if (!ready)
               pFunc = PyObject_GetAttrString(pModule, pythonModule); // pFunc is a new reference

          if (pFunc && PyCallable_Check(pFunc)) {
               pArgs = PyTuple_New(valuesForTest.size());
               for (int i = 0; i < valuesForTest.size(); ++i) {
                   pValue = PyFloat_FromDouble(valuesForTest.at(i));
                   if (!pValue) {
                       Py_DECREF(pArgs);
                       Py_DECREF(pModule);
                       fprintf(stderr, "Cannot convert argument\n");
                       pythonReady = false;
                   }
                   /* pValue reference stolen here: */
                   PyTuple_SetItem(pArgs, i, pValue);
               }
               pValue = PyObject_CallObject(pFunc, pArgs);
               Py_DECREF(pArgs);
               if (pValue != NULL) {
                   valuesForResult = listTupleToVector_Float(pValue);
                   if (valuesForResult.size() > 0 ){
                         //cout << "Result of call :"; 
                         //for (int i = 0; i < valuesForResult.size(); i++)
                         //     cout << valuesForResult.at(i) << " ";
                         //cout << endl;
                         Py_DECREF(pValue);
                         pythonReady = true;
                   }
                   else 
                       return EXIT_FAILURE;
               }
               else {
                   Py_DECREF(pFunc);
                   Py_DECREF(pModule);
                   PyErr_Print();
                   fprintf(stderr,"Call failed\n");
                   pythonReady = false;
               }
           }
           else {
               if (PyErr_Occurred())
                   PyErr_Print();
               fprintf(stderr, "Cannot find function \"%s\"\n", pythonModule);
           }
        }
        else {
           PyErr_Print();
           fprintf(stderr, "Failed to load \"%s\"\n", pythonScript);
           pythonReady = false;
        }
        return pythonReady;
}

void closePythonStuff() {
     Py_DECREF(pName);
     Py_XDECREF(pFunc);
     Py_DECREF(pModule);
}

//Define a real-valued solution vector, called fgObjectiveVector
typedef moeoRealObjectiveVector <moeoObjectiveVectorTraits> fgObjectiveVector; // 

//Structure of the genotype (i.e. solution vector) for the foamingGlassDesign problem. vector of doubles, objectivefuncVal1, objectivefuncVal2
class foamingGlassDesign : public moeoRealVector <fgObjectiveVector, double, double> {};

//Operator for the evaluation of the objective function
class foamingGlassEval : public moeoEvalFunc <foamingGlassDesign> {
public:

     /**
     * operator evaluates a genotype (i.e., a vector of double)
     */
     void operator () (foamingGlassDesign & _element){
          if (_element.invalidObjectiveVector()){
               /** NOTE: Keep this an example of an O.F: computed directly in C++
                  int nbFun= foamingGlassDesign::ObjectiveVector::nObjectives();
                  int nbVar= _element.size();
                  int k;
                  double g;
                  fgObjectiveVector objVec;
                  k = nbVar - nbFun + 1;
                  g = 0.0;
                  
                  for (unsigned i = nbVar - k + 1; i <= nbVar; i++)
                      g += pow(_element[i-1]-0.5,2) - cos(20 * M_PI * (_element[i-1]-0.5));
                  g = 100 *(k + g);

                  for (unsigned i = 1; i <= nbFun; i++) {
                      double f = 0.5 * (1 + g);
                      for (unsigned j = nbFun - i; j >= 1; j--)
                          f *= _element[j-1];
                      if (i > 1)
                          f *= 1 - _element[(nbFun - i + 1) - 1];
                      objVec[i-1] = f;
                  }
                  _element.objectiveVector(objVec); **/
               
               std::vector<double> valuesForResult;
               fgObjectiveVector objVec; //Bi-objective vector
               PyObject *pArgs, *pValue;
               
               pArgs = PyTuple_New(_element.size());
               
               for (int i = 0; i < _element.size(); ++i) {
                   pValue = PyFloat_FromDouble(_element.at(i));
                   PyTuple_SetItem(pArgs, i, pValue);
               }
               pValue = PyObject_CallObject(pFunc, pArgs);
               Py_DECREF(pArgs);
               
               //Keeps this verification to avoid unexpected errors
               if (pValue != NULL) {
                    valuesForResult = listTupleToVector_Float(pValue);
                    
                    //If the vector returns 4 values, the second and fourth positions are the semi standard deviation
                    //and we use the semi standard deviation to penalize the predicted values. The ideas of using the
                    //semi standard deviation is to penalize only solution that have higher dispersion towards negative
                    //quality, and not the other way around
                    
                    if (valuesForResult.size() > 2){
                         //" penalized "; 
                         int j=0;
                         for (int i = 0; i < valuesForResult.size(); i++){
                              if (i%2==0){
                                   objVec[j] = ((1-penalty_weight)*valuesForResult.at(i)) - 
                                                  (penalty_weight*valuesForResult.at(i+1));
                                   j++;
                              }
                         }
                   }
                   else {
                         //"not penalized"; 
                         for (int i = 0; i < valuesForResult.size(); i++)
                              objVec[i] = valuesForResult.at(i);
                   }
                    //Get ready for the next F.E.";
                    Py_DECREF(pValue);
                     _element.objectiveVector(objVec);
                     //Give the user some indication of what is going on
                     num_FE_used++;
                     if (num_FE_used%10==0 && num_FE_used < max_FE)
                          cout << "\n   FE used so far " << num_FE_used << "";
               }
               else {
                    Py_DECREF(pArgs);
                    closePythonStuff();
                    PyErr_Print();
                    fprintf(stderr,"Call failed\n");
                    exit (-1);
               }
          } //invalidObjectiveVector
     } //operator
}; //class

int main(int argc, char* argv[]) {
     try {
    
          eoParser parser(argc, argv);  // for user-parameter reading
          eoState state;                // to keep all things allocated

          // Parameters
          unsigned int MAX_GEN = parser.createParam((unsigned int)(10000), "maxGen", "Maximum number of generations",'G',"Param").value();
          double P_CROSS = parser.createParam(1.0, "pCross", "Crossover probability",'C',"Param").value();
          double EXT_P_MUT = parser.createParam(1.0, "extPMut", "External Mutation probability",'E',"Param").value();
          double INT_P_MUT = parser.createParam(0.083, "intPMut", "Internal Mutation probability",'I',"Param").value();
          unsigned int VEC_SIZE = parser.createParam((unsigned int)(12), "vecSize", "Genotype Size",'V',"Param").value();
          unsigned int NB_OBJ= parser.createParam((unsigned int)(3), "nbObj", "Number of Objective",'N',"Param").value();
          std::string OUTPUT_FILE = parser.createParam(std::string("dtlz_ibea"), "outputFile", "Path of the output file",'o',"Output").value();
          unsigned int EVAL = parser.createParam((unsigned int)(1), "eval", "Number of the DTLZ evaluation fonction",'F',"Param").value();
          unsigned int DTLZ4_PARAM = parser.createParam((unsigned int)(100), "dtlz4_param", "Parameter of the DTLZ4 evaluation fonction",'P',"Param").value();
          unsigned int NB_EVAL = parser.createParam((unsigned int)(0), "nbEval", "Number of evaluation before Stop",'P',"Param").value();
          unsigned int TIME = parser.createParam((unsigned int)(0), "time", "Time(seconds) before Stop",'T',"Param").value();

          // Create an initialize the min/max vector of objectives (true = min / false = max)
          std::vector <bool> bObjectives(NB_OBJ);
          bObjectives[0]=true;
          bObjectives[1]=false;
          moeoObjectiveVectorTraits::setup(NB_OBJ,bObjectives); 
          
          // Initialize python to be able to query the model (i.e. the o.f.)
          if (!initPythonStuff(false))
               exit (-1);
          max_FE = NB_EVAL; //this is only for printing when evaluating the o.f.
          //initPythonStuff(true); //It can also be run like these with better safety-guards, but this over-killing it

          // The fitness function evaluation
          eoEvalFunc <foamingGlassDesign> * eval;
          eval= new foamingGlassEval;

          // Vector of doubles to constraint the value of the variables of the problem
          std::vector<double> lowBound;
          std::vector<double> upBound;
          //waterglasscontent (0, 30)
          lowBound.push_back(0);
          upBound.push_back(30);
          //N330 (0.0, 1.0)
          lowBound.push_back(0);
          upBound.push_back(1);
          //K3PO4 (0, 4)
          lowBound.push_back(0);
          upBound.push_back(4);
          //Mn3O4 (0, 7) 
          lowBound.push_back(0);
          upBound.push_back(7);
          //drying (NO, YES)
          lowBound.push_back(0);
          upBound.push_back(1);
          //mixing (CLASSICAL, ADDITIONAL)
          lowBound.push_back(0);
          upBound.push_back(1);
          //furnace_temperature (700, 805)
          lowBound.push_back(700);
          upBound.push_back(805);
          //heating_rate (1.0, 5.0)
          lowBound.push_back(1);
          upBound.push_back(5);
          //foaming_time (5.0, 60)
          lowBound.push_back(5);
          upBound.push_back(60);

          // Define a solution through an initializer 
          eoRealVectorBounds bounds(lowBound,upBound);
          //eoRealVectorBounds bounds(VEC_SIZE, 0.0, 1.0);
          eoRealInitBounded <foamingGlassDesign> init (bounds);
          
          // Variation operators
          SBXCrossover <foamingGlassDesign> xover(bounds, 15);
          PolynomialMutation <foamingGlassDesign> mutation(bounds, INT_P_MUT, 20);

          // Stopping criteria
          eoGenContinue <foamingGlassDesign> term(MAX_GEN);
          eoEvalFuncCounter <foamingGlassDesign> evalFunc(*eval);
          
          eoCheckPoint <foamingGlassDesign> *checkpoint;
          if (TIME > 0)
               checkpoint = new eoCheckPoint <foamingGlassDesign> (*(new eoTimeContinue< foamingGlassDesign >(TIME)));
          else if (NB_EVAL > 0)
               checkpoint = new eoCheckPoint <foamingGlassDesign> (*(new eoEvalContinue< foamingGlassDesign >(evalFunc, NB_EVAL)));
          else {
               cout << "ERROR!!! : TIME or NB_EVAL must be > 0 : used option --time or --nbEval\n";
               return EXIT_FAILURE;
          }
          checkpoint->add(term);

          /*moeoArchiveObjectiveVectorSavingUpdater <foamingGlassDesign> updater(arch, OUTPUT_FILE);
          checkpoint->add(updater);*/

          // Build the algorithm via its components
          eoSGAGenOp <foamingGlassDesign> op(xover, P_CROSS, mutation, EXT_P_MUT);
          moeoAdditiveEpsilonBinaryMetric <fgObjectiveVector> metric;
          moeoIBEA <foamingGlassDesign> algo(*checkpoint, evalFunc ,op, metric);


          /*** Go ! ***/
          // Create the initial population
          eoPop <foamingGlassDesign> &pop = do_make_pop(parser, state, init);

          // Need help?
          make_help(parser);

          // Run the algo
          do_run(algo, pop);

          // Extract first front of the final population using an moeoArchive (this is the output of IBEA)
          moeoUnboundedArchive <foamingGlassDesign> finalArchive;
          finalArchive(pop);

          // Printing of the final population
          ofstream outfile(OUTPUT_FILE.c_str(), ios::app);
          ofstream outfileFull((OUTPUT_FILE +"Full").c_str(), ios::app);
          outfileFull << "\nPredicted_Density_[Kg/m^3] Apparent_Closed_Porosity[/] num_components waterglasscontent_[0,30] N330_[0.0,1.0] K3PO4_[0,4] Mn3O4_[0,7] drying_[NO,YES] mixing_[CLASSICAL,ADDITIONAL] furnace_temperature_[700,805] heating_rate_[1.0,5.0] foaming_time_[5.0,60]\n";
          finalArchive.sortedPrintOn(outfileFull);
          outfileFull.close();

          for (unsigned int i=0 ; i < finalArchive.size(); i++) {
               for (unsigned int j=0 ; j<NB_OBJ; j++) {
                    outfile << finalArchive[i].objectiveVector()[j];
                    if (j != NB_OBJ -1)
                         outfile << " ";
               }
               outfile << endl;
          }
          outfile.close();
          cout << endl;
     }
     catch (exception& e) {
          cout << e.what() << endl;
     }
     return EXIT_SUCCESS;
}
