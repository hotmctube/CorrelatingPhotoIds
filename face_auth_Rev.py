'''
Script to analyse discrepancies from official photo documentation and actual photo
Author: manojc
'''
import pandas as pd
import ast
import seaborn as sns
import matplotlib.pyplot as plt

# Clean up dates with YYYY-MM format
def transform_date(x):
    y = f'01/{x[-2:]}/{x[:4]}'
    return str(y)

def isNaN(num):
    return num != num

def transform_result(x):
    if x == 'clear':
        return 3
    else:
        return 0

def transform_sub_result(x):
    if x == 'clear':
        return 3
    elif x == 'caution':
        return 2
    elif x == 'suspected':
        return 1
    else:
        return 0

def transform_gender(x):
    if x == 'Female':
        return 2
    elif x == 'Male':
        return 1
    else:
        return 0

def transform_enums(items, x):
    indexPos = items.index(x)
    return indexPos+1

def numerate_cols(col):
    if col == 'issuing_country' or col == 'nationality':
        unique_enums = unique_country_enums
    else:
        unique_enums = corr_df[col].unique().tolist()

    unique_enums = [x for x in unique_enums if str(x) != 'nan' and str(x) != 'None']
    corr_df[col] = corr_df[col].apply(lambda x: int(transform_enums(unique_enums,str(x))) if len(str(x)) > 0 and not isNaN(x) else 0 )

def plot_heatmap(gen_corr_df, out_file, title_header):
    ax = plt.axes()
    mat = gen_corr_df.corr(method='pearson')
    mat.to_csv(out_file, index=None, header=True)
    sns.heatmap(mat, vmin=mat.min().min(), vmax=mat.max().max(), fmt="", cmap='RdYlGn', linewidths=0.30, ax=ax)
    ax.set_title(title_header)
    plt.show()


doc_data_file = "C:\\Rev\doc_reports_sample.csv"
doc_data_out_file = "C:\\Rev\doc_reports_out.csv"
face_data_file = "C:\\Rev\\face_reports_sample.csv"
face_data_out_file = "C:\\Rev\\face_reports_out.csv"
merged_data_out_file = "C:\\Rev\\merged_reports_out.csv"

corr_all_data_out_file = "C:\\Rev\\corr_all_data_out.csv"
corr_doc_data_out_file = "C:\\Rev\\corr_doc_data_out.csv"
corr_face_data_out_file = "C:\\Rev\\corr_face_data_out.csv"

doc_df = pd.read_csv(doc_data_file)
face_df = pd.read_csv(face_data_file)

print(len(doc_df))
print(len(face_df))

doc_df['properties'] = doc_df['properties'].apply(ast.literal_eval)
doc_df = pd.concat([doc_df.drop(['properties'], axis=1), doc_df['properties'].apply(pd.Series)], axis=1)

face_df['properties'] = face_df['properties'].apply(ast.literal_eval)
face_df = pd.concat([face_df.drop(['properties'], axis=1), face_df['properties'].apply(pd.Series)], axis=1)
face_df['score'] = face_df['score'].apply(lambda x: x if len(str(x)) > 0 else 0)

doc_df['issuing_date'] = doc_df['issuing_date'].apply(lambda x: transform_date(str(x)) if (len(str(x))) == 7 else x)
doc_cols = list(doc_df)

doc_cols = list(map(lambda x: x if x != 'result' else 'doc_result', doc_cols))
doc_cols = list(map(lambda x: x if x != 'visual_authenticity_result' else 'doc_visual_authenticity_result', doc_cols))

doc_df.columns = doc_cols

merged_df = pd.merge(doc_df,
                   face_df[['attempt_id','result', 'face_comparison_result', 'facial_image_integrity_result', 'visual_authenticity_result', 'score']],
                   how='left',
                   on='attempt_id')


merged_cols = list(merged_df)
merged_cols = list(map(lambda x: x if x != 'result' else 'face_result', merged_cols))
merged_df.columns = merged_cols

corr_df = merged_df.copy()

bool_col_list = ['doc_result','doc_visual_authenticity_result',	'image_integrity_result', 'face_detection_result', 'image_quality_result', 'supported_document_result',	'conclusive_document_quality_result', 'colour_picture_result', 'data_validation_result', 'data_consistency_result',	'data_comparison_result', 'police_record_result', 'compromised_document_result', 'face_result', 'face_comparison_result', 'facial_image_integrity_result', 'visual_authenticity_result']
enum_col_list = ['document_type', 'issuing_country', 'nationality', 'issuing_state']

for col in bool_col_list:
    corr_df[col] = corr_df[col].apply(lambda x: int(transform_result(str(x))))

corr_df['sub_result'] = corr_df['sub_result'].apply(lambda x: int(transform_sub_result(str(x))))
corr_df['gender'] = corr_df['gender'].apply(lambda x: int(transform_gender(str(x))))

unique_country_enums = corr_df['issuing_country'].unique().tolist() + corr_df['nationality'].unique().tolist()
unique_set = set(unique_country_enums)
unique_country_enums = list(unique_set)

for col in enum_col_list:
    numerate_cols(col)

del corr_df['Unnamed: 0']
del corr_df['user_id']
del corr_df['attempt_id']


corr_df_doc = corr_df[ corr_df['doc_result'] <= 0]
corr_df_face = corr_df[ corr_df['face_result'] <= 0]

corr_df_doc.to_csv(merged_data_out_file,index=None, header=True)

plot_heatmap(corr_df, corr_all_data_out_file, 'Overall Document and Facial Image Success and Failure Heatmap')
plot_heatmap(corr_df_doc, corr_doc_data_out_file, 'Document Failure Heatmap')
plot_heatmap(corr_df_face, corr_face_data_out_file, 'Facial Image Failure Heatmap')

