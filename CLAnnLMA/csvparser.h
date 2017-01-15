#ifndef CSVPARSER_H
#define CSVPARSER_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CsvRow {
    char **fields_;
    int numOfFields_;
} CsvRow;

typedef struct CsvParser {
    char *filePath_;
    char delimiter_;
    int firstLineIsHeader_;
    char *errMsg_;
    CsvRow *header_;
    FILE *fileHandler_;
	int fromString_;
	char *csvString_;
	int csvStringIter_;
} CsvParser;


// Public
CsvParser *CsvParser_new(const char *filePath, const char *delimiter, int firstLineIsHeader);
CsvParser *CsvParser_new_from_string(const char *csvString, const char *delimiter, int firstLineIsHeader);
void CsvParser_destroy(CsvParser *csvParser);
void CsvParser_destroy_row(CsvRow *csvRow);
CsvRow *CsvParser_getHeader(CsvParser *csvParser);
CsvRow *CsvParser_getRow(CsvParser *csvParser);
int CsvParser_getNumFields(CsvRow *csvRow);
char **CsvParser_getFields(CsvRow *csvRow);
const char* CsvParser_getErrorMessage(CsvParser *csvParser);

// Private
CsvRow *_CsvParser_getRow(CsvParser *csvParser);    
int _CsvParser_delimiterIsAccepted(const char *delimiter);
void _CsvParser_setErrorMessage(CsvParser *csvParser, const char *errorMessage);

#ifdef __cplusplus
}
#endif

#endif


//int levmarq(int npar, double *par, int ny, double *y, double *dysq,
//			double (*func)(double *, int, void *),
//			void (*grad)(double *, double *, int, void *),
//			void *fdata)
//{
//	int x,i,j,it,nit,ill,verbose;
//	double lambda,up,down,mult,weight,err,newerr,derr,target_derr;
//	double h[npar][npar],ch[npar][npar];
//	double g[npar],d[npar],delta[npar],newpar[npar];
//
//	//PAR ChiQuadro
//
//	/* main iteration */
//	for (it=0; it<nit; it++) {
//
//		//PAR JACOBIAN & HESSIAN
//
//		mult = 1 + lambda;
//		ill = 1; /* ill-conditioned? */
//		while (ill && (it<nit)) {
//
//			//CHOLESKY DECOMPOSITION
//
//			if (!ill) {
//				//CHOLESKY SOLVE
//
//				//NEW PAR update with delta
//				//NEW PAR ChiQuadro
//				derr = newerr - err;
//				ill = (derr > 0);
//			}
//
//			if (ill) {
//				mult = (1 + lambda*up)/(1 + lambda);
//				lambda *= up;
//				it++;
//			}
//		}
//		//PAR update with NEW PAR
//		err = newerr;
//		lambda *= down;
//
//		if ((!ill)&&(-derr<target_derr)) break;
//	}
//
//	return (it==nit);
//}