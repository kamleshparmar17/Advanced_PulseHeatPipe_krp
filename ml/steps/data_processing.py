# data selection from cleaned experimental data and prepare for further data analysis and ML

import pandas as pd
import sys, os
from typing import Annotated, Tuple
from zenml import step
from PyPulseHeatPipe import PulseHeatPipe
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser

class DataProcessingEngine:
    """
    Data Processing Engine is used for various data processing related steps

        loading meta table

        slicing data as per meta table and make a experimental database

        removing garbage data from failed experiments, threshold 1000

        calculating TR and other Thermal params with the help of PyPulseHeatPipe library
    
    """
    def __init__(self, dir_path: str):
        """
        to initialize data processing engine

        args:
            dir_path: str '../data/'

        returns:
            None
        """
        self.dir_path = dir_path

    def load_meta_table(self, file_name: str='meta_table_data.csv'):
        """
        to load meta table that contain information of the all experimental data

        args:
            file_name: str

        returns:
            df: pd.DataFrame
        """
        meta_table_path = self.dir_path + file_name
        df = pd.read_csv(meta_table_path)

        # basic cleaning
        df.dropna(thresh=9, inplace=True)
        df.drop
        df.drop_duplicates(inplace=True)
        return df
    
    def processing_date_time(self, df:pd.DataFrame, 
                             col:str='date',
                             col_date:str='DATE',
                             col_time:str='TIME')-> Annotated[pd.DataFrame,'Process DateTime col']:
        """
        to process date time cols from row data and returns timestamp col for further analysis

        args:
            df:pd.DataFrame # experimental raw data
            col:str # col name
            format:str # datetime expected format
            col_date:str='DATE',
            col_time:str='TIME'

        use:
            df = processing_date_time(df, col, format)

        """
        # Define a function to parse dates using dateutil.parser
        def parse_date(date_str):
            try:
                #dayfirst=True to handle day/month/year format correctly
                return parser.parse(date_str, dayfirst=True)
            except ValueError:
                return None
            
        df[col] = df[col_date].astype(str) + ' ' + df[col_time].astype(str)
        df[col] = df[col].apply(parse_date)
        return df

    
    def data_slicing_combine(self, 
                            df_meta: pd.DataFrame, 
                            df_raw_data: pd.DataFrame,
                            col_start: str='dt_start',
                            col_stop: str='dt_stop',
                            max_search_seconds: int=60):
        """
        Slice data from the clean experimental combined data based on column values.

        Args:
            df_meta: pd.DataFrame containing metadata
            df_raw_data: pd.DataFrame containing raw data
            col_start: str, column name for experiment start time (default: 'dt_start')
            col_stop: str, column name for experiment stop time (default: 'dt_stop')
            max_search_seconds: int, maximum seconds to incrementally search (default: 60)

        Returns:
            pd.DataFrame: combined DataFrame
        """
        frames = []
        for _, row in df_meta.iterrows():
            # Convert experiment start and stop timestamps to datetime objects
            experiment_start = pd.to_datetime(row[col_start], format='%d-%m-%Y%H:%M:%S')
            experiment_stop = pd.to_datetime(row[col_stop], format='%d-%m-%Y%H:%M:%S')
            
            # Try to find closest match by incrementally searching up to max_search_seconds
            found_start = False
            found_stop = False
            increment = timedelta(seconds=1)
            search_time = timedelta(seconds=max_search_seconds)
            
            while not (found_start and found_stop) and search_time >= timedelta(seconds=0):
                # Check if experiment_start is found
                if any(df_raw_data['date'] == experiment_start):
                    found_start = True
                else:
                    experiment_start += increment
                
                # Check if experiment_stop is found
                if any(df_raw_data['date'] == experiment_stop):
                    found_stop = True
                else:
                    experiment_stop += increment
                
                # Decrease the remaining search time
                search_time -= increment
            
            # Slice raw data based on experiment start and stop times
            df_sd = df_raw_data[(df_raw_data['date'] >= experiment_start) & (df_raw_data['date'] <= experiment_stop)].copy()
            print(f"Timestamps {experiment_start} or {experiment_stop} found in df_raw_data index.")
    
            # Add metadata columns to sliced data
            df_sd['WF'] = row['WF']
            df_sd['FR[%]'] = row['FR [%]']
            df_sd['Q[W]'] = row['Q [W]']
            df_sd['alpha'] = row['alpha']
            df_sd['beta'] = row['beta']
            df_sd['pulse'] = row['t_pulse_start']
            
            frames.append(df_sd)
        
        # Concatenate all frames into a single DataFrame
        df_database = pd.concat(frames, ignore_index=True)

        return df_database
    
    def database_to_csv(self, df_database:pd.DataFrame, op_path:str):
        """
        to save database to csv (local)

        args:
            df_database: pd.DataFrame
            op_path: str

        returns:
            csv
        """
        # Directory and file path
        directory = os.path.join(op_path, 'database')
        file_path = os.path.join(op_path, directory, 'database.csv')

        # Check if directory exists and create it if not
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the DataFrame to a CSV file in the specified directory
        df_database.to_csv(file_path, index=False)
        return None
    
    def adding_stat_cols(self, df_database:pd.DataFrame, col_c:list=['TC_6','TC_7','TC_8', 'TC_9'], 
                                                col_e:list=['TC_1', 'TC_2', 'TC_3', 'TC_4', 'TC_5']):
        """
        adding mean and std cols for temperature

        args:
            df_database:pd.DataFrame, 
            col_c:list=['TC_6','TC_7','TC_8', 'TC_9'], 
            col_e:list=['TC_1', 'TC_2', 'TC_3', 'TC_4', 'TC_5']
            
        returns:
            pd.DataFrame
        """
        df = df_database
        df['Te_mean[C]'] = df[col_e].mean(axis=1)
        df['Tc_mean[C]'] = df[col_c].mean(axis=1)
        df['Te_std[C]'] = df[col_e].std(axis=1)
        df['Tc_std[C]'] = df[col_c].std(axis=1)

        return df
    
    def removing_garbage_data(self, 
                              df_database:pd.DataFrame,
                              threshold:int=1000):
        """
        dropping experimental garbage data. dropping datasets with less then 1000 entries 

        args:
            df_database: pd.DataFrame
            threshold: int = 1000
        
        returns:
            df: pd.DataFrame
        """
        # Calculate value counts for 'Q[W]' and 'FR[%]'
        df_values = df_database[['Q[W]', 'FR[%]']].value_counts().reset_index(name='count')
        
        # Filter out the values below the threshold
        values_to_drop = df_values[df_values['count'] < threshold]
        
        # Initialize the filtered dataframe
        filtered_df = df_database.copy()
        
        # Drop the rows where 'Q[W]' and 'FR[%]' match the values below the threshold
        for _, row in values_to_drop.iterrows():
            q = row['Q[W]']
            fr = row['FR[%]']
            indices_to_drop = filtered_df[(filtered_df['Q[W]'] == q) & (filtered_df['FR[%]'] == fr)].index
            filtered_df = filtered_df.drop(indices_to_drop)
            filtered_df.fillna(0, inplace=True)
        
        return filtered_df
    
    def get_si_units(self,
                     database:pd.DataFrame,
                     sample:str = 'DI_Water'):
        """
        Convert to SI units

        args:
            database:pd.DataFrame,
            sample:str = 'DI_Water'

        returns:
            pd.DataFrame
        """
        analysis = PulseHeatPipe(dir_path=self.dir_path, sample=sample)
        data = analysis.convert_to_si(df=database)
        return data

    def get_absolute_pressure(self,
                              database: pd.DataFrame,
                              P_out_col: str = 'P[bar]',
                              P_col:str = 'PRESSURE'
                              ):
        """
        to calculate absolute pressure

        args:
            database: pd.DataFrame,
            P_out_col: str = 'P[bar]'
            P_col: str

        returns:
            pd.DataFrame
        """
        database[P_out_col] = database[P_col] + 1.013
        return database

    def get_thermal_resistance(self,
                               data:pd.DataFrame,
                               T_e: str = 'Te_mean[K]',
                               T_c: str = 'Tc_mean[K]',
                               Q_c: str = 'Q[W]',
                               sample: str = 'DI_Water',
                               TR_out: str = 'TR[K/W]'):
        """
        to calculate thermal resistance

        args:
            data:pd.DataFrame,
            T_e: str = 'Te_mean[K]',
            T_c: str = 'Tc_mean[K]',
            Q_c: str = 'Q[W]'
            sample: str = 'DI_Water',
            TR_out: str = 'TR[K/W]'
            
        returns:
            pd.DataFrame

        """
        analysis = PulseHeatPipe(dir_path=self.dir_path, sample=sample)
        data = analysis.compute_thermal_resistance(data=data,
                                                   T_condenser_col=T_c,
                                                   T_evaporator_col=T_e,
                                                   Q_heat_col=Q_c,
                                                   TR_output_col=TR_out)
        return data

    def get_gfe(self,
                database: pd.DataFrame,
                sample: str = 'DI_Water'):
        """
        to calculate the gibbs free energy

        args:
            database: pd.DataFrame,
            sample: str = 'DI_Water'

        returns:
            pd.DataFrame
        """
        analysis = PulseHeatPipe(dir_path=self.dir_path, sample=sample)
        database = analysis.compute_gibbs_free_energy(data=database,
                                                      T_evaporator_col='Te_mean[K]',
                                                      T_condenser_col='Tc_mean[K]',
                                                      P_bar='P[bar]',
                                                      to_csv=False)
        return database
    
    def get_pulse_temperature(self,
                              database: pd.DataFrame):
        '''
        to get a temperature at which pulsation starts

        args:
            database: pd.DataFrame

        returns:
            pd.DataFrame
        '''
        database['TIME'] = pd.to_datetime(database['TIME'], format='%H:%M:%S').dt.time

        frames = []
        pulse_time = database['pulse'].unique()
        for time in pulse_time:
            df_pt = database[database['TIME']==time]
            Te_pt = df_pt['Te_mean[K]'].min()
            db = database[database['pulse']==time]
            db['T_pulse[K]'] = Te_pt
            frames.append(db)
        db = pd.concat(frames, axis=0, ignore_index=True)
        return db

    
@step
def step_initialize_DPE(dir_path:str)->Annotated[DataProcessingEngine, 'Data Processing Engine']:
    dpe = DataProcessingEngine(dir_path=dir_path)
    return dpe


@step
def step_loading_meta_table(dpe:DataProcessingEngine, filename:str = 'meta_table_data.csv')->Annotated[pd.DataFrame, 'Meta Table Loaded']:
    df_meta = dpe.load_meta_table(file_name=filename)
    return df_meta

@step
def step_meta_data_dt_process(dpe:DataProcessingEngine, 
                              df_meta:pd.DataFrame)->Annotated[pd.DataFrame, 'Meta Table - DT Processed']:
    df_meta_processed = dpe.processing_date_time(df=df_meta, col='dt_start', col_date='Date', col_time='t_start')
    df_meta_processed = dpe.processing_date_time(df=df_meta, col='dt_stop', col_date='Date', col_time='t_end')
    return df_meta_processed

@step
def step_database(dpe:DataProcessingEngine,
                  df_meta:pd.DataFrame,
                  df_raw:pd.DataFrame)->Annotated[pd.DataFrame, 'Experimental DataBase']:
    df_database = dpe.data_slicing_combine(df_meta=df_meta, df_raw_data=df_raw, col_start='dt_start', col_stop='dt_stop')
    return df_database

@step
def step_processing_dt_col_pulse(dpe:DataProcessingEngine,
                                 df_database:pd.DataFrame)->Annotated[pd.DataFrame, 'Experimental DataBase with Pulse Col']:
    database = dpe.processing_date_time(df=df_database,
                                           col='pulse',
                                           col_date='DATE',
                                           col_time='pulse')
    database['pulse'] = database['pulse'].dt.time
    return database

@step
def step_stat_cols(dpe:DataProcessingEngine,
                   df_database:pd.DataFrame)-> Annotated[pd.DataFrame, 'DataBase with Statistical Features']:
    df_databse_ = dpe.adding_stat_cols(df_database=df_database)
    return df_databse_

@step
def step_dropping_garbage_date(dpe:DataProcessingEngine,
                               df_database:pd.DataFrame)->Annotated[pd.DataFrame, 'Removing Garbage']:
    df_database_f = dpe.removing_garbage_data(df_database=df_database)
    return df_database_f

@step
def step_to_abs_pressure(dpe:DataProcessingEngine,
                         database:pd.DataFrame)->Annotated[pd.DataFrame, 'Added Absolute Pressure']:
    df_database = dpe.get_absolute_pressure(database=database, P_out_col='P[bar]', P_col='PRESSURE')
    return df_database

@step
def step_to_si_units(dpe:DataProcessingEngine,
                     database:pd.DataFrame)->Annotated[pd.DataFrame, 'Converted to SI Units']:
    df_database = dpe.get_si_units(database=database, sample='DI_Water')
    return df_database

@step
def step_adding_pulse_temp(dpe:DataProcessingEngine,
                                 df_database:pd.DataFrame)->Annotated[pd.DataFrame, 'Experimental DataBase with Pulse Temperature']:
    database = dpe.get_pulse_temperature(database=df_database)
    return database

@step
def step_TR_calculation(dpe:DataProcessingEngine,
                        database:pd.DataFrame)->Annotated[pd.DataFrame, 'Calculating Thermal Resistance']:
    df_database = dpe.get_thermal_resistance(data=database)
    return df_database

@step
def step_gfe_calculation(dpe:DataProcessingEngine,
                         database:pd.DataFrame)->Annotated[pd.DataFrame, 'Calculating Gibbs Free Energy']:
    df_database = dpe.get_gfe(database=database, sample='DI_Water')
    return df_database

@step
def step_database_csv(dpe:DataProcessingEngine,
                      df_database:pd.DataFrame)->Annotated[None, 'Saved CSV locally']:
    df_database = df_database.round(2)
    dpe.database_to_csv(df_database=df_database, op_path=dpe.dir_path)
    return None