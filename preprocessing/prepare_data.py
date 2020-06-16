# -*- coding: utf-8 -*-
import preprocessing
import numpy as np


pp = preprocessing.Preprocess(data_path_c='dict2csv/variants.csv',read_dtype={'motif.ehipos':object, 'motif.ename':object, 'cadd.istv':object},autoEliminateNullColumns=False,autoImpute=False) #https://stackoverflow.com/a/27232309/8149411
data = pp.getData()
data = data.astype({'motif.ehipos':np.bool, 'motif.ename':np.bool, 'cadd.istv':str})
pp.dropCols(['_id','cadd._license','clinvar._license','clinvar.rsid','_score'])
pp.eliminateNullColumns()
pp.impute()
data = pp.getData()
data['cadd.isderived'] = data['cadd.isderived'] == 1
data['cadd.istv'] = data['cadd.istv'] == 'TRUE'



data = data[data['rcv.clinical_significance'] != 'Uncertain significance']
data = data[data['rcv.clinical_significance'] != 'not provided']
data = data[data['rcv.clinical_significance'] != 'other']
data = data[data['rcv.clinical_significance'] != 'risk factor']
data = data[data['rcv.clinical_significance'] != 'Conflicting interpretations of pathogenicity']
data = data[data['rcv.clinical_significance'] != 'Benign/Likely benign']
data = data[data['rcv.clinical_significance'] != 'Pathogenic/Likely pathogenic']

#data = data[data['rcv.clinical_significance'] == 'Uncertain significance'] #only VUS

data = data.reset_index(drop=True) # Index sıfırlanmadığı zaman. İteration'lar eski size'e göre dönüyor. yani verinin bir kısmını işlemiyor.

pp.setData(data)
data = pp.getData()

columns = []
for i in data:
    columns.append([i, data[i].dtypes.name])

def write_encoded(data):    
    pp.encode()
    data = pp.getData()
    data.to_csv('variants_encoded.csv', index=False)
    return data
    

def write_not_encoded(data):
    data.to_csv('variants_not_encoded.csv', index=False)
    
write_not_encoded(data)
#
data = write_encoded(data)

print(data.info())


