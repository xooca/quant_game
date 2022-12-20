import duckdb
import pandas as pd
import numpy as np
from datetime import datetime

def create_connection(db_path,args_dict={'read_only':False}):
    args_dict['database'] = db_path
    return duckdb.connect(**args_dict)

def write_df_to_table_with_date(connection,table_name,df,as_of_dt=None):
    if as_of_dt is None:
        as_of_dt = datetime.today().strftime('%Y-%m-%d')
        df['as_of_dt'] == as_of_dt
    count = connection.execute(f"SELECT count(*) FROM {df} where as_of_dt = {as_of_dt} limit 5")
    if count > 0 and len(df) > 0:
        connection.execute(f"DELETE from {table_name} where as_of_dt = {as_of_dt}")
    if len(df) > 0:
        connection.execute(f"INSERT INTO {table_name} SELECT * FROM {df}")

def write_df_to_table(connection,table_name,df,create_table=False):
    if create_table:
        connection.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {df}")
    connection.execute(f"INSERT INTO {table_name} SELECT * FROM {df}")

def load_table_df(connection,table_name):
    return connection.execute(f"select * from {table_name}").df()

def check_if_table_exists(connection,table_name):
    info_df = load_table_df(connection,'information_schema.tables')
    info_df['table_name_schema'] = info_df['table_schema'] + '.' + info_df['table_name']
    if table_name in set(info_df['table_name_schema'].to_list()):
        return True
    else:
        return False

def check_if_table_and_column_exists(connection,table_name,column_name):
    info_df = load_table_df(connection,'information_schema.columns')
    info_df['table_name_schema'] = info_df['table_schema'] + '.' + info_df['table_name']
    info_df = info_df[info_df['table_name_schema']==table_name]
    if len(info_df) > 0 and column_name in set(info_df['column_name'].to_list()):
        return True
    else:
        return False

def execute_sql(connection,sql):
    connection.execute(sql)

def join_table(connection,table1,table2,table1_on,table2_on,how='left outer',where_condition=None):
    sql = f"select table1.*,table2.* from {table1} as table {how} {table2} as table2 on table1.{table1_on}=table2.{table2_on}"
    if where_condition is not None:
        sql = sql + ' ' + f"where {where_condition}"
    print(f"join sql is {sql}")
    df = connection.execute(sql).df()
    return df

def alter_table(connection,table_name,alter_arg):
    if alter_arg.get('alter_type')=='add_column':
        sql = f"ALTER TABLE {table_name} ADD COLUMN {alter_arg.get('column_name')} {alter_arg.get('data_type')} "   
    elif alter_arg.get('alter_type')=='drop_column':
        sql = f"ALTER TABLE {table_name} DROP {alter_arg['column_name']}"
    elif alter_arg.get('alter_type')=='change_datatype':
        sql = f"ALTER TABLE {table_name} ALTER {alter_arg['column_name']} TYPE {alter_arg.get('data_type')}"
    elif alter_arg.get('alter_type')=='change_default':
        sql = f"ALTER TABLE {table_name} ALTER COLUMN {alter_arg['column_name']} SET {alter_arg.get('default_value')}"
    elif alter_arg.get('alter_type')=='drop_default':
        sql = f"ALTER TABLE {table_name} ALTER COLUMN {alter_arg['column_name']} DROP DEFAULT"
    elif alter_arg.get('alter_type')=='rename_table':
        sql = f"ALTER TABLE {table_name} RENAME TO {alter_arg['new_table_name']}"
    elif alter_arg.get('alter_type')=='rename_column':
        sql = f"ALTER TABLE {table_name} RENAME {alter_arg['column_name']} TO {alter_arg['new_column_name']}"
    else:
        print("choose correct alter_type")
    print(f"alter sql is {sql}")
    connection.execute(sql)


def update_table(connection,table_name,update_arg):
    sql = f'''UPDATE {table_name} SET {update_arg.get('column_name')} = {update_arg.get('set_expr')} WHERE 
    {update_arg.get('where_expr')}'''
    print(f"update sql is {sql}")
    connection.execute(sql)

def delete_data(connection,table_name,delete_arg,df=None):
    if 'using_expr' in delete_arg:
        sql = f'''
        DELETE FROM {table_name} USING {delete_arg.get('using_expr')} WHERE {delete_arg.get('column_name')} = {delete_arg.get('delete_expr')}  
        '''
    else:
        sql = f'''
        DELETE FROM {table_name} WHERE {delete_arg.get('column_name')} = {delete_arg.get('delete_expr')}  
        '''
    if df is not None and 'column_name' in delete_arg and 'select_condition_column_name' in delete_arg:
        sql = f'''
        DELETE FROM {table_name} WHERE {delete_arg.get('column_name')} in select {delete_arg.get('select_condition_column_name')}
        from df  
        ''' 
    print(f"DELETE sql is {sql}")
    connection.execute(sql)

def insert_data(connection,table_name,insert_arg,df=None):
    if 'inserting_table' in insert_arg:
        sql = f'''
        INSERT INTO {table_name} 
        SELECT {insert_arg.get('inserting_table_columns')} from {insert_arg.get('inserting_table')}
        '''
    
    if 'inserting_sql' in insert_arg:
        sql = f'''
        INSERT INTO {table_name} {insert_arg.get('inserting_sql')}
        ''' 

    if 'insert_values' in insert_arg:
        sql = f'''
        INSERT INTO {table_name} VALUES {insert_arg.get('insert_values')}
        ''' 
        if 'insert_column' in insert_arg:
            sql = f'''
            INSERT INTO {table_name}({insert_arg.get('insert_column')}) VALUES {insert_arg.get('insert_values')}
            '''
        
    if df is not None and 'inserting_table_columns' in insert_arg:
        sql = f'''
        INSERT INTO {table_name} 
        SELECT {insert_arg.get('inserting_table_columns')} from df
        '''
        if 'inserting_where_clause' in insert_arg:
            sql = f'''
            INSERT INTO {table_name} 
            SELECT {insert_arg.get('inserting_table_columns')} from df
            where {insert_arg.get('inserting_where_clause')}
            '''

    if df is not None and 'inserting_table_columns' not in insert_arg:
        sql = f'''
        INSERT INTO {table_name} 
        SELECT * from df
        '''
        if 'inserting_where_clause'  in insert_arg:
            sql = f'''
            INSERT INTO {table_name} 
            SELECT * from df where {insert_arg.get('inserting_where_clause')}
            '''
    print(f"INSERT sql is {sql}")
    connection.execute(sql)

def create_schema(connection,schema_name):
    sql = f"CREATE SCHEMA IF NOT EXISTS {schema_name}"
    print(f"create schema sql is {sql}")
    connection.execute(sql)

def create_table(connection,table_name,create_table_arg,df=None):
    if 'select_sql' in create_table_arg and (create_table_arg.get('replace')==False or create_table_arg.get('replace') is None):
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} AS {create_table_arg.get('select_sql')}"
    if 'table_column_arg' in create_table_arg and (create_table_arg.get('replace')==False or create_table_arg.get('replace') is None):
        sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({create_table_arg.get('table_column_arg')})"
    if 'select_sql' in create_table_arg and create_table_arg.get('replace')==True:
        sql = f"CREATE OR REPLACE TABLE {table_name} AS {create_table_arg.get('select_sql')}"
    if 'table_column_arg' in create_table_arg and create_table_arg.get('replace')==True:
        sql = f"CREATE OR REPLACE TABLE {table_name} ({create_table_arg.get('table_column_arg')})"  
    if  df is not None and create_table_arg.get('replace')==True:
        sql = f"CREATE OR REPLACE TABLE {table_name} AS select * from df"  
    if  df is not None and create_table_arg.get('replace')==False:
        sql = f"CREATE TABLE {table_name} AS select * from df" 
    print(f"create table sql is {sql}")
    connection.execute(sql)

def create_view(connection,view_name,create_view_arg):
    if (create_view_arg.get('replace')==False or create_view_arg.get('replace') is None):
        if 'select_sql' in create_view_arg:
            sql = f"CREATE OR REPLACE VIEW {view_name} AS {create_view_arg.get('select_sql')}"
        if 'select_sql' in create_view_arg:
            sql = f"CREATE OR REPLACE VIEW {view_name} AS {create_view_arg.get('select_sql')}"
        if ('select_sql' in create_view_arg and 'view_column_name' in create_view_arg):
            sql = f"CREATE OR REPLACE VIEW {view_name}({create_view_arg.get('view_column_arg')}) AS {create_view_arg.get('select_sql')}"

    else:
        if 'select_sql' in create_view_arg:
            sql = f"CREATE VIEW {view_name} AS {create_view_arg.get('select_sql')}"
        if 'select_sql' in create_view_arg:
            sql = f"CREATE VIEW {view_name} AS {create_view_arg.get('select_sql')}"
        if ('select_sql' in create_view_arg and 'view_column_name' in create_view_arg):
            sql = f"CREATE VIEW {view_name}({create_view_arg.get('view_column_arg')}) AS {create_view_arg.get('select_sql')}"
  
    print(f"create view sql is {sql}")
    connection.execute(sql)

def drop_object(connection,object_type,object_name):
    sql = f"DROP {object_type} IF EXISTS {object_name}"
    print(f"DROP sql is {sql}")
    connection.execute(sql)