# -*- coding: utf-8 -*-
import preprocessing
import numpy as np

pp = preprocessing.Preprocess(data_path_c='dict2csv/variants.csv',read_dtype={'motif.ehipos':object, 'motif.ename':object, 'cadd.istv':object},autoEliminateNullColumns=False,autoImpute=False) #https://stackoverflow.com/a/27232309/8149411
data = pp.getData()
data = data.astype({'motif.ehipos':np.bool, 'motif.ename':np.bool, 'cadd.istv':str})
pp.dropCols(['_id','cadd._license','clinvar._license','clinvar.rsid','_score'])
pp.eliminateNullColumns()
pp.impute()
pp.encode()
data = pp.getData()
data.to_csv('variants_encoded.csv', index=False)
#data.to_csv('variants_not_encoded.csv', index=False)

