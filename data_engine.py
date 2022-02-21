from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import datetime as dt
from talib.abstract import *
import zipfile,fnmatch,os
import pandas as pd
import pickle
from pathlib import Path
from feature_engine.discretisation import EqualWidthDiscretiser
from feature_engine.imputation import MeanMedianImputer,CategoricalImputer,ArbitraryNumberImputer,EndTailImputer,DropMissingData

def unzip_folders(rootPath,pattern):
    for root, dirs, files in os.walk(rootPath):
        for filename in fnmatch.filter(files, pattern):
            print(os.path.join(root, filename))
            zipfile.ZipFile(os.path.join(root, filename)).extractall(os.path.join(root, os.path.splitext(filename)[0]))
            os.remove(os.path.join(root, filename))

def convert_df_to_timeseries(df):
    df['date_time'] = df['date'].astype(str) + ' ' + df['time']
    df = df.sort_values(by='date_time')
    df.index = df['date_time']
    df = df[['open','high','low','close']]
    return df

def create_dataset(root_path,pattern,data_save_path,data_name,reset_df = False):
    files_list = []
    bad_files = []
    files_processed = []
    base_df = pd.DataFrame(columns = ['name','date','time','open','high','low','close'])
    if not os.path.exists(f'{data_save_path}{data_name}/'):
        os.makedirs(f'{data_save_path}{data_name}/')
        print(f'Created folder {data_save_path}{data_name}')

    already_loaded_file_name = f'{data_save_path}{data_name}/already_loaded_files.pickle'
    csv_save_location = f'{data_save_path}{data_name}/{data_name}.csv'
    print(f'Data save path is {csv_save_location}')
    print(f'File with already loaded files is {already_loaded_file_name}')
    orig_cols = ['name','date','time','open','high','low','close']
    try:
        with open(already_loaded_file_name, 'rb') as handle:
            already_loaded_files = pickle.load(handle)
            already_loaded_files = [Path(col) for col in already_loaded_files]
            print(f"Total files already saved {len(already_loaded_files)}")
    except Exception as e1:
        print(f"File {already_loaded_file_name} is not loaded because of error : {e1}")
        already_loaded_files = []
    for root, dirs, files in os.walk(root_path):
        for filename in fnmatch.filter(files, pattern):
            f_name = Path(os.path.join(root, filename))
            files_list.append(f_name)
    files_to_be_loaded = [f for f in files_list if f not in already_loaded_files]
    files_to_be_loaded = list(dict.fromkeys(files_to_be_loaded))
    files_list = list(dict.fromkeys(files_list))
    print(f"Total files detected {len(files_list)}")
    print(f"Total new files detected {len(files_to_be_loaded)}")
    try:
        base_df = pd.read_csv(csv_save_location)
    except Exception as e1:
        print(f"Error while loading dataframe from {csv_save_location} because of error : {e1}")
        base_df = pd.DataFrame(columns = ['open','high','low','close'])
        files_to_be_loaded = files_list
    if len(base_df) == 0 or reset_df:
        files_to_be_loaded = files_list
        print(f"We are going to reload all the data")

    print(f"Number of files to be loaded {len(files_to_be_loaded)}")
    base_df_st_shape = base_df.shape
    for i,f_name in enumerate(files_to_be_loaded,1):
        f_name = os.path.join(root, f_name)
        try:
            tmp_df = pd.read_csv(f_name,header=None)
            tmp_df = tmp_df.loc[:,0:6]
            tmp_df.columns = orig_cols
            tmp_df = convert_df_to_timeseries(tmp_df)
            base_df = pd.concat([base_df,tmp_df],axis=0)
            print(len(files_to_be_loaded)-i,base_df.shape,f_name)
            already_loaded_files.append(f_name)
        except Exception as e1:
            bad_files.append(f_name)
            print(f"File {f_name} is not loaded because of error : {e1}")
    with open(already_loaded_file_name, 'wb') as handle:
        pickle.dump(already_loaded_files, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Shape of the dataframe before duplicate drop is {base_df.shape}")
    base_df = base_df.drop_duplicates()
    print(f"Shape of the dataframe after duplicate drop is {base_df.shape}")
    if base_df_st_shape != base_df.shape:
        base_df = base_df.sort_index()
        base_df.to_csv(csv_save_location, index_label=False )
        print(f"Saving dataframe to location {csv_save_location}")
    return base_df

class Signals:
    def __init__(self, df):
        self.df = df
        self.high = np.array(self.df['high'].astype('float'))
        self.low = np.array(self.df['low'].astype('float'))
        self.open = np.array(self.df['open'].astype('float'))
        self.close = np.array(self.df['close'].astype('float'))

    def momentum_ADX(self, timeperiod=14):
        self.df['momentum_ADX'] = ADX(
            self.high, self.low, self.close, timeperiod)

    def momentum_ADXR(self, timeperiod=14):
        self.df['momentum_ADXR'] = ADXR(
            self.high, self.low, self.close, timeperiod)

    def momentum_APO(self, fastperiod=12, slowperiod=26, matype=0):
        self.df['momentum_APO'] = APO(
            self.close, fastperiod, slowperiod, matype)

    def momentum_AROON(self, timeperiod=14):
        self.df['momentum_AROON_DOWN'], self.df['momentum_AROON_UP'] = AROON(
            self.high, self.low, self.close, timeperiod)

    def momentum_AROONOSC(self, timeperiod=14):
        self.df['momentum_AROONOSC'] = AROONOSC(
            self.high, self.low, self.close, timeperiod)

    def momentum_BOP(self, timeperiod=14):
        self.df['momentum_BOP'] = BOP(
            self.open, self.high, self.low, self.close, timeperiod)

    def momentum_CCI(self, timeperiod=14):
        self.df['momentum_CCI'] = CCI(
            self.high, self.low, self.close, timeperiod)

    def momentum_CMO(self, timeperiod=14):
        self.df['momentum_CMO'] = CMO(self.close, timeperiod)

    def momentum_DX(self, timeperiod=14):
        self.df['momentum_DX'] = DX(
            self.high, self.low, self.close, timeperiod)

    def momentum_MACD(self, fastperiod=12, slowperiod=26, signalperiod=9):
        self.df['momentum_MACD'], self.df['momentum_MACD_SIGNAL'], self.df['momentum_MACD_HIST'] = MACD(
            self.close, fastperiod, slowperiod, signalperiod)

    def momentum_MACDEXT(self, fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0):
        self.df['momentum_MACDEXT'], self.df['momentum_MACDEXT_SIGNAL'], self.df['momentum_MACDEXT_HIST'] = MACDEXT(
            self.close, fastperiod, fastmatype, slowperiod, slowmatype, signalperiod, signalmatype)

    def momentum_MACDFIX(self, signalperiod=9):
        self.df['momentum_MACDFIX'], self.df['momentum_MACDFIX_SIGNAL'], self.df['momentum_MACDFIX_HIST'] = MACDFIX(
            self.close, signalperiod)

    def momentum_MFI(self, timeperiod=14):
        self.df['momentum_MFI'] = MFI(
            self.high, self.low, self.close, self.volume, timeperiod)

    def momentum_MINUS_DM(self, timeperiod=14):
        self.df['momentum_MINUS_DM'] = MINUS_DM(
            self.high, self.low, timeperiod)

    def momentum_MOM(self, timeperiod=10):
        self.df['momentum_MOM'] = MOM(self.low, timeperiod)

    def momentum_PLUS_DI(self, timeperiod=14):
        self.df['momentum_PLUS_DI'] = PLUS_DI(
            self.high, self.low, self.close, timeperiod)

    def momentum_PLUS_DM(self, timeperiod=14):
        self.df['momentum_PLUS_DM'] = PLUS_DM(self.high, self.low, timeperiod)

    def momentum_PPO(self, fastperiod=12, slowperiod=26, matype=0):
        self.df['momentum_PPO'] = PPO(
            self.close, fastperiod, slowperiod, matype)

    def momentum_ROC(self, timeperiod=10):
        self.df['momentum_ROC'] = ROC(self.close, timeperiod)

    def momentum_ROCP(self, timeperiod=10):
        self.df['momentum_ROCP'] = ROCP(self.close, timeperiod)

    def momentum_ROCR(self, timeperiod=10):
        self.df['momentum_ROCR'] = ROCR(self.close, timeperiod)

    def momentum_ROCR100(self, timeperiod=10):
        self.df['momentum_ROCR100'] = ROCR100(self.close, timeperiod)

    def momentum_RSI(self, timeperiod=14):
        self.df['momentum_RSI'] = RSI(self.close, timeperiod)

    def momentum_STOCH(self, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0):
        self.df['momentum_SLOWK'], self.df['momentum_SLOWD'] = STOCH(
            self.high, self.low, self.close, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)

    def momentum_STOCHF(self, fastk_period=5, fastd_period=3, fastd_matype=0):
        self.df['momentum_FASTK'], self.df['momentum_FASTD'] = STOCHF(
            self.high, self.low, self.close, fastk_period, fastd_period, fastd_matype)

    def momentum_STOCHRSI(self, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0):
        self.df['momentum_STOCHRSI_FASTK'], self.df['momentum_STOCHRSI_FASTD'] = STOCHRSI(
            self.close, timeperiod, fastk_period, fastd_period, fastd_matype)

    def momentum_TRIX(self, timeperiod=30):
        self.df['momentum_TRIX'] = TRIX(self.close, timeperiod)

    def momentum_ULTOSC(self, timeperiod1=7, timeperiod2=14, timeperiod3=28):
        self.df['momentum_ULTOSC'] = ULTOSC(
            self.high, self.low, self.close, timeperiod1, timeperiod2, timeperiod3)

    def momentum_WILLR(self, timeperiod=14):
        self.df['momentum_WILLR'] = WILLR(
            self.high, self.low, self.close, timeperiod)

    def volumn_AD(self):
        self.df['volume_AD'] = AD(self.high, self.low, self.close, self.volume)

    def volumn_ADOSC(self, fastperiod=3, slowperiod=10):
        self.df['volume_ADOSC'] = ADOSC(
            self.high, self.low, self.close, self.volume, fastperiod, slowperiod)

    def volumn_OBV(self):
        self.df['volume_OBV'] = OBV(
            self.high, self.low, self.close, self.volume)

    def volatile_ATR(self, timeperiod=14):
        self.df['volatile_ATR'] = ATR(
            self.high, self.low, self.close, timeperiod)

    def volatile_NATR(self, timeperiod=14):
        self.df['volatile_NATR'] = NATR(
            self.high, self.low, self.close, timeperiod)

    def volatile_TRANGE(self):
        self.df['volatile_TRANGE'] = TRANGE(self.high, self.low, self.close)

    def transform_AVGPRICE(self):
        self.df['transform_AVGPRICE'] = AVGPRICE(
            self.open, self.high, self.low, self.close)

    def transform_MEDPRICE(self):
        self.df['transform_MEDPRICE'] = MEDPRICE(self.high, self.low)

    def transform_TYPPRICE(self):
        self.df['transform_TYPPRICE'] = TYPPRICE(
            self.high, self.low, self.close)

    def transform_WCLPRICE(self):
        self.df['transform_WCLPRICE'] = WCLPRICE(
            self.high, self.low, self.close)

    def cycle_HT_DCPERIOD(self):
        self.df['cycle_HT_DCPERIOD'] = HT_DCPERIOD(self.close)

    def cycle_HT_DCPHASE(self):
        self.df['cycle_HT_DCPHASE'] = HT_DCPHASE(self.close)

    def cycle_HT_PHASOR(self):
        self.df['cycle_HT_PHASOR_inphase'], self.df['cycle_HT_PHASOR_quadrature'] = HT_PHASOR(
            self.close)

    def cycle_HT_SINE(self):
        self.df['cycle_HT_SINE'], self.df['cycle_HT_SINE_LEAD'] = HT_SINE(
            self.close)

    def cycle_HT_TRENDMODE(self):
        self.df['cycle_HT_TRENDMODE'] = HT_TRENDMODE(self.close)

    def pattern_2_crows(self):
        self.df['pattern_2_crows'] = CDL2CROWS(
            self.open, self.high, self.low, self.close)

    def pattern_3_black_crows(self):
        self.df['pattern_3_black_crows'] = CDL3BLACKCROWS(
            self.open, self.high, self.low, self.close)

    def pattern_3_inside_updown(self):
        self.df['pattern_3_inside_updown'] = CDL3INSIDE(
            self.open, self.high, self.low, self.close)

    def pattern_3_line_strike(self):
        self.df['pattern_3_line_strike'] = CDL3LINESTRIKE(
            self.open, self.high, self.low, self.close)

    def pattern_3_outside_updown(self):
        self.df['pattern_3_outside_updown'] = CDL3OUTSIDE(
            self.open, self.high, self.low, self.close)

    def pattern_3_stars_south(self):
        self.df['pattern_3_stars_south'] = CDL3STARSINSOUTH(
            self.open, self.high, self.low, self.close)

    def pattern_3_adv_white_soldier(self):
        self.df['pattern_3_adv_white_soldier'] = CDL3WHITESOLDIERS(
            self.open, self.high, self.low, self.close)

    def pattern_abondoned_baby(self, penetration=0):
        self.df['pattern_abondoned_baby'] = CDLABANDONEDBABY(
            self.open, self.high, self.low, self.close, penetration)

    def pattern_advance_block(self):
        self.df['pattern_advance_block'] = CDLADVANCEBLOCK(
            self.open, self.high, self.low, self.close)

    def pattern_belt_hold(self):
        self.df['pattern_belt_hold'] = CDLBELTHOLD(
            self.open, self.high, self.low, self.close)

    def pattern_breakaway(self):
        self.df['pattern_breakaway'] = CDLBREAKAWAY(
            self.open, self.high, self.low, self.close)

    def pattern_closing_marubozu(self):
        self.df['pattern_closing_marubozu'] = CDLCLOSINGMARUBOZU(
            self.open, self.high, self.low, self.close)

    def pattern_concealing_baby_swallow(self):
        self.df['pattern_concealing_baby_swallow '] = CDLCONCEALBABYSWALL(
            self.open, self.high, self.low, self.close)

    def pattern_counter_attack(self):
        self.df['pattern_counter_attack '] = CDLCOUNTERATTACK(
            self.open, self.high, self.low, self.close)

    def pattern_dark_cloud_cover(self, penetration=0):
        self.df['pattern_dark_cloud_cover '] = CDLDARKCLOUDCOVER(
            self.open, self.high, self.low, self.close, penetration)

    def pattern_doji(self):
        self.df['pattern_doji '] = CDLDOJI(
            self.open, self.high, self.low, self.close)

    def pattern_doji_star(self):
        self.df['pattern_doji_star '] = CDLDOJISTAR(
            self.open, self.high, self.low, self.close)

    def pattern_dragonfly_doji(self):
        self.df['pattern_dragonfly_doji '] = CDLDRAGONFLYDOJI(
            self.open, self.high, self.low, self.close)

    def pattern_engulfing_pattern(self):
        self.df['pattern_engulfing_pattern '] = CDLENGULFING(
            self.open, self.high, self.low, self.close)

    def pattern_evening_doji_star(self):
        self.df['pattern_evening_doji_star'] = CDLEVENINGDOJISTAR(
            self.open, self.high, self.low, self.close)

    def pattern_evening_star(self):
        self.df['pattern_evening_star'] = CDLEVENINGSTAR(
            self.open, self.high, self.low, self.close)

    def pattern_updown_gapside_white_lines(self):
        self.df['pattern_updown_gapside_white_lines'] = CDLGAPSIDESIDEWHITE(
            self.open, self.high, self.low, self.close)

    def pattern_gravestone_doji(self):
        self.df['pattern_gravestone_doji'] = CDLGRAVESTONEDOJI(
            self.open, self.high, self.low, self.close)

    def pattern_hammer(self):
        self.df['pattern_hammer'] = CDLHAMMER(
            self.open, self.high, self.low, self.close)

    def pattern_hanging_man(self):
        self.df['pattern_hanging_man'] = CDLHANGINGMAN(
            self.open, self.high, self.low, self.close)

    def pattern_harami(self):
        self.df['pattern_harami'] = CDLHARAMI(
            self.open, self.high, self.low, self.close)

    def pattern_harami_cross(self):
        self.df['pattern_harami_cross'] = CDLHARAMICROSS(
            self.open, self.high, self.low, self.close)

    def pattern_high_wave_candle(self):
        self.df['pattern_high_wave_candle'] = CDLHIGHWAVE(
            self.open, self.high, self.low, self.close)

    def pattern_hikkake(self):
        self.df['pattern_hikkake'] = CDLHIKKAKE(
            self.open, self.high, self.low, self.close)

    def pattern_modified_hikkake(self):
        self.df['pattern_modified_hikkake'] = CDLHIKKAKEMOD(
            self.open, self.high, self.low, self.close)

    def pattern_homing_pigeon(self):
        self.df['pattern_homing_pigeon'] = CDLHOMINGPIGEON(
            self.open, self.high, self.low, self.close)

    def pattern_identical_3_cross(self):
        self.df['pattern_harami_cross'] = CDLIDENTICAL3CROWS(
            self.open, self.high, self.low, self.close)

    def pattern_in_neck(self):
        self.df['pattern_in_neck'] = CDLINNECK(
            self.open, self.high, self.low, self.close)

    def pattern_inverted_hammer(self):
        self.df['pattern_inverted_hammer'] = CDLINVERTEDHAMMER(
            self.open, self.high, self.low, self.close)

    def pattern_kicking(self):
        self.df['pattern_kicking'] = CDLKICKING(
            self.open, self.high, self.low, self.close)

    def pattern_kicking_bullbear_morubozu(self):
        self.df['pattern_kicking_bullbear_morubozu'] = CDLKICKINGBYLENGTH(
            self.open, self.high, self.low, self.close)

    def pattern_ladder_bottom(self):
        self.df['pattern_ladder_bottom'] = CDLLADDERBOTTOM(
            self.open, self.high, self.low, self.close)

    def pattern_long_leg_doji(self):
        self.df['pattern_long_leg_doji'] = CDLLONGLEGGEDDOJI(
            self.open, self.high, self.low, self.close)

    def pattern_long_line_candle(self):
        self.df['pattern_long_line_candle'] = CDLLONGLINE(
            self.open, self.high, self.low, self.close)

    def pattern_morubozu(self):
        self.df['pattern_morubozu'] = CDLMARUBOZU(
            self.open, self.high, self.low, self.close)

    def pattern_matching_low(self):
        self.df['pattern_matching_low'] = CDLMATCHINGLOW(
            self.open, self.high, self.low, self.close)

    def pattern_mat_holding(self):
        self.df['pattern_mat_holding'] = CDLMATHOLD(
            self.open, self.high, self.low, self.close)

    def pattern_morning_doji_star(self, penetration=0):
        self.df['pattern_morning_doji_star'] = CDLMORNINGDOJISTAR(
            self.open, self.high, self.low, self.close, penetration)

    def pattern_morning_star(self, penetration=0):
        self.df['pattern_morning_star'] = CDLMORNINGSTAR(
            self.open, self.high, self.low, self.close, penetration)

    def pattern_on_neck(self):
        self.df['pattern_on_neck'] = CDLONNECK(
            self.open, self.high, self.low, self.close)

    def pattern_piercing(self):
        self.df['pattern_piercing'] = CDLPIERCING(
            self.open, self.high, self.low, self.close)

    def pattern_rickshaw_man(self):
        self.df['pattern_rickshaw_man'] = CDLRICKSHAWMAN(
            self.open, self.high, self.low, self.close)

    def pattern_risefall_3_methods(self):
        self.df['pattern_rickshaw_man'] = CDLRISEFALL3METHODS(
            self.open, self.high, self.low, self.close)

    def pattern_separating_lines(self):
        self.df['pattern_separating_lines'] = CDLSEPARATINGLINES(
            self.open, self.high, self.low, self.close)

    def pattern_shooting_star(self):
        self.df['pattern_shooting_star'] = CDLSHOOTINGSTAR(
            self.open, self.high, self.low, self.close)

    def pattern_short_line(self):
        self.df['pattern_short_line'] = CDLSHORTLINE(
            self.open, self.high, self.low, self.close)

    def pattern_spinning_top(self):
        self.df['pattern_spinning_top'] = CDLSPINNINGTOP(
            self.open, self.high, self.low, self.close)

    def pattern_stalled(self):
        self.df['pattern_stalled'] = CDLSTALLEDPATTERN(
            self.open, self.high, self.low, self.close)

    def pattern_stick_sandwich(self):
        self.df['pattern_stick_sandwich'] = CDLSTICKSANDWICH(
            self.open, self.high, self.low, self.close)

    def pattern_takuri(self):
        self.df['pattern_takuri'] = CDLTAKURI(
            self.open, self.high, self.low, self.close)

    def pattern_tasuki_gap(self):
        self.df['pattern_tasuki_gap'] = CDLTASUKIGAP(
            self.open, self.high, self.low, self.close)

    def pattern_thrusting_pattern(self):
        self.df['pattern_thrusting_pattern'] = CDLTHRUSTING(
            self.open, self.high, self.low, self.close)

    def pattern_tristar_pattern(self):
        self.df['pattern_tristar_pattern'] = CDLTRISTAR(
            self.open, self.high, self.low, self.close)

    def pattern_unique_3_river(self):
        self.df['pattern_unique_3_river'] = CDLUNIQUE3RIVER(
            self.open, self.high, self.low, self.close)

    def pattern_upside_gap_2_crows(self):
        self.df['pattern_upside_gap_2_crows'] = CDLUPSIDEGAP2CROWS(
            self.open, self.high, self.low, self.close)

    def pattern_updown_gap_3_method(self):
        self.df['pattern_updown_gap_3_method'] = CDLXSIDEGAP3METHODS(
            self.open, self.high, self.low, self.close)

    def stats_beta(self, timeperiod=5):
        self.df['stats_beta'] = BETA(self.high, self.low, timeperiod)

    def stats_pearson_coeff(self, timeperiod=30):
        self.df['stats_pearson_coeff'] = CORREL(
            self.high, self.low, timeperiod)

    def stats_linear_reg(self, timeperiod=14):
        self.df['stats_linear_reg'] = LINEARREG(self.close, timeperiod)

    def stats_linear_reg_angle(self, timeperiod=14):
        self.df['stats_linear_reg_angle'] = LINEARREG_ANGLE(
            self.close, timeperiod)

    def stats_linear_reg_intercept(self, timeperiod=14):
        self.df['stats_linear_reg_intercept'] = LINEARREG_INTERCEPT(
            self.close, timeperiod)

    def stats_linear_reg_slope(self, timeperiod=14):
        self.df['stats_linear_reg_slope'] = LINEARREG_SLOPE(
            self.close, timeperiod)

    def stats_linear_reg_slope(self, timeperiod=5, nbdev=1):
        self.df['stats_linear_reg_slope'] = STDDEV(
            self.close, timeperiod, nbdev)

    def stats_time_series_forecast(self, timeperiod=14):
        self.df['stats_linear_reg_slope'] = TSF(self.close, timeperiod)

    def stats_variance(self, timeperiod=5, nbdev=1):
        self.df['stats_variance'] = VAR(self.close, timeperiod, nbdev)

    def math_ACOS(self):
        self.df['math_ACOS'] = ACOS(self.close)

    def math_ASIN(self):
        self.df['math_ASIN'] = ASIN(self.close)

    def math_ATAN(self):
        self.df['math_ATAN'] = ATAN(self.close)

    def math_CEIL(self):
        self.df['math_CEIL'] = CEIL(self.close)

    def math_COS(self):
        self.df['math_COS'] = COS(self.close)

    def math_COSH(self):
        self.df['math_COSH'] = COSH(self.close)

    def math_EXP(self):
        self.df['math_EXP'] = EXP(self.close)

    def math_FLOOR(self):
        self.df['math_FLOOR'] = FLOOR(self.close)

    def math_LN(self):
        self.df['math_LN'] = LN(self.close)

    def math_LOG10(self):
        self.df['math_LOG10'] = LOG10(self.close)

    def math_SIN(self):
        self.df['math_SIN'] = SIN(self.close)

    def math_SINH(self):
        self.df['math_SINH'] = SINH(self.close)

    def math_SQRT(self):
        self.df['math_SQRT'] = SQRT(self.close)

    def math_TAN(self):
        self.df['math_TAN'] = TAN(self.close)

    def math_TANH(self):
        self.df['math_TANH'] = TANH(self.close)

    def overlap_bolliner_bands(self, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
        self.df['overlap_bb_upper'], self.df['overlap_bb_middle'], self.df['overlap_bb_lower'] = BBANDS(
            self.close, timeperiod, nbdevup, nbdevdn, matype)

    def overlap_double_exp_moving_avg(self, timeperiod=30):
        self.df['overlap_double_exp_moving_avg'] = DEMA(self.close, timeperiod)

    def overlap_exp_moving_avg(self, timeperiod=30):
        self.df['overlap_exp_moving_avg'] = EMA(self.close, timeperiod)

    def overlap_hilbert_transform(self):
        self.df['overlap_hilbert_transform'] = HT_TRENDLINE(self.close)

    def overlap_kaufman_adaptive_moving_avg(self, timeperiod=30):
        self.df['overlap_kaufman_adaptive_moving_avg'] = KAMA(
            self.close, timeperiod)

    def overlap_moving_avg(self, timeperiod=30, matype=0):
        self.df['overlap_moving_avg'] = MA(self.close, timeperiod, matype)

    def overlap_mesa_adaptive_moving_average(self, fastlimit=0, slowlimit=0):
        self.df['overlap_mama'], self.df['overlap_fama'] = MAMA(
            self.close, fastlimit, slowlimit)

    def overlap_moving_avg_with_variable(self, periods, minperiod=2, maxperiod=30, matype=0):
        self.df['overlap_moving_avg_with_variable'] = MAVP(
            self.close, periods, minperiod, maxperiod, matype)

    def overlap_midpoint(self, timeperiod=14):
        self.df['overlap_midpoint'] = MIDPOINT(self.close, timeperiod)

    def overlap_midpoint_price(self, timeperiod=14):
        self.df['overlap_midpoint_price'] = MIDPRICE(
            self.high, self.low, timeperiod)

    def overlap_parabolic_sar(self, acceleration=0, maximum=0):
        self.df['overlap_parabolic_sar'] = SAR(
            self.high, self.low, acceleration, maximum)

    def overlap_parabolic_sar_ext(self, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0):
        self.df['overlap_parabolic_sar_ext'] = SAREXT(self.high, self.low, startvalue, offsetonreverse, accelerationinitlong,
                                                      accelerationlong, accelerationmaxlong, accelerationinitshort, accelerationshort, accelerationmaxshort)

    def overlap_simple_moving_avg(self, timeperiod=30):
        self.df['overlap_simple_moving_avg'] = SMA(self.close, timeperiod)

    def overlap_triple_exp_moving_average_t3(self, timeperiod=5, vfactor=0):
        self.df['overlap_triple_exp_moving_average_t3'] = T3(
            self.close, timeperiod, vfactor)

    def overlap_triangular_moving_average(self, timeperiod=30):
        self.df['overlap_triangular_moving_average'] = TRIMA(
            self.close, timeperiod)

    def overlap_weighted_moving_average(self, timeperiod=30):
        self.df['overlap_weighted_moving_average'] = WMA(
            self.close, timeperiod)

class LabelCreator(BaseEstimator, TransformerMixin):
    def __init__(self, freq='1min',shift=-15,shift_column='close'):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.label_name = f'label_{shift}_{freq}_{shift_column}'
        
    def fit(self, X, y=None):
        return self    # Nothing to do in fit in this scenario
    
    def label_generator(self,val):
        if val <= 10 and val>=10:
            return '-10to10'
        elif val > 10 and val <= 20:
            return '10to20'
        elif val > 20 and val <= 40:
            return '20to40'
        elif val > 40 and val <= 60:
            return '40to60'
        elif val > 60 and val <= 80:
            return '60to80'
        elif val > 80 and val <= 100:
            return '80to100'
        elif val > 100:
            return 'above100'
        elif val < -10 and val >= -20:
            return '-10to-20'
        elif val < -20 and val >= 40:
            return '-20to-40'
        elif val < -40 and val >= 60:
            return '-40to-60'
        elif val < -60 and val >= 80:
            return '-60to-80'
        elif val < -80 and val >= 100:
            return '-80to-100'
        elif val < 100:
            return 'below100'
        else:
            return 'unknown'

    def transform(self, df):
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df[self.label_name] = df.shift(self.shift, freq=self.freq)[self.shift_column].subtract(df[self.shift_column]).apply(self.label_generator)  
        print(f"Shape of dataframe after transform is {df.shape}") 
        return df

class TechnicalIndicator(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.all_methods = []
        a = dict(Signals.__dict__)
        for a1,a2 in a.items():
            self.all_methods.append(a1)
        self.all_methods = [m1 for m1,m2 in a.items() if m1[:1]!='_']
        
    def fit(self, X, y=None):
        return self    # Nothing to do in fit in this scenario
    
    def transform(self, df):
        sig = Signals(df)
        self.methods_run = []
        self.methods_notrun = []
        for f in self.all_methods:
            try:
                exec(f'sig.{f}()')
                self.methods_run.append(f)
            except Exception as e1:
                print(f"Function {f} was unable to run, Error is {e1}")
                self.methods_notrun.append(f)
        print(f"Shape of dataframe after transform is {df.shape}")
        return sig.df

class NormalizeDataset(BaseEstimator, TransformerMixin):
    def __init__(self, columns = [],impute_values=False,impute_type = 'categorical',convert_to_floats = False,arbitrary_impute_variable=99,drop_na_col=False,drop_na_rows=False,
    fillna = False,fillna_method = 'bfill'):
        self.impute_values = impute_values
        self.convert_to_floats = convert_to_floats
        self.columns = columns
        self.impute_type = impute_type
        self.arbitrary_impute_variable = arbitrary_impute_variable
        self.drop_na_col = drop_na_col
        self.drop_na_rows = drop_na_rows
        self.fillna_method = fillna_method
        self.fillna = fillna

    def fit(self, X, y=None):
        return self    # Nothing to do in fit in this scenario

    def transform(self, df):
        df = convert_todate_deduplicate(df)
        if self.convert_to_floats:
            for col in self.columns:
                df[col] = df[col].astype('float')
        if self.impute_values:

            from sklearn.pipeline import Pipeline
            if self.impute_type == 'mean_median_imputer':
                imputer = MeanMedianImputer(imputation_method='median', variables=self.columns)
            elif self.impute_type == 'categorical':
                imputer = CategoricalImputer(variables=self.columns)
            elif self.impute_type == 'arbitrary':
                if isinstance(self.arbitrary_impute_variable, dict):
                    imputer = ArbitraryNumberImputer(imputer_dict = self.arbitrary_impute_variable)
                else:
                    imputer = ArbitraryNumberImputer(variables = self.columns,arbitrary_number = self.arbitrary_number)
            else:
                imputer = CategoricalImputer(variables=self.columns)
            imputer.fit(df)
            df= imputer.transform(df)
        if self.fillna:
            df = df.fillna(method=self.fillna_method)
        if self.drop_na_col:
            imputer = DropMissingData(missing_only=True)
            imputer.fit(df)
            df= imputer.transform(df)
        if self.drop_na_rows:
            #df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
            df = df.dropna(axis=0)
        print(f"Shape of dataframe after transform is {df.shape}")
        return df
class LastTicksGreaterValuesCount(BaseEstimator, TransformerMixin):
    def __init__(self, columns,create_new_col = True,last_ticks=10):
        self.columns = columns
        self.last_ticks = last_ticks
        self.create_new_col = create_new_col
        
    def fit(self, X, y=None):
        return self    # Nothing to do in fit in this scenario
    
    def rolling_window(self,a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def transform(self, df):
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]         
        for col in self.columns:
            x = np.concatenate([[np.nan] * (self.last_ticks), df[col].values])
            arr = self.rolling_window(x, self.last_ticks + 1)
            if self.create_new_col:
                #df[f'last_tick_{col}_{self.last_ticks}'] = self.rolling_window(x, self.#last_ticks + 1)
                df[f'last_tick_{col}_{self.last_ticks}']  = (arr[:, :-1] > arr[:, [-1]]).sum(axis=1)
            else:
                df[col] = (arr[:, :-1] > arr[:, [-1]]).sum(axis=1)
        print(f"Shape of dataframe after transform is {df.shape}")
        return df

def convert_todate_deduplicate(df):
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')] 
        return df

class PriceLastTickBreachCount(BaseEstimator, TransformerMixin):
    def __init__(self, columns,create_new_col = True,last_ticks='10min',breach_type = ['mean']):
        self.columns = columns
        self.last_ticks = last_ticks
        self.create_new_col = create_new_col
        self.breach_type = breach_type
        
    def fit(self, X, y=None):
        return self    # Nothing to do in fit in this scenario

    def transform(self, df):
        #df.index = pd.to_datetime(df.index)
        #df = df.sort_index()
        #df = df[~df.index.duplicated(keep='first')]         
        for col in self.columns:
            for breach_type in self.breach_type:
                if self.create_new_col:
                    col_name = f'last_tick_{breach_type}_{col}_{self.last_ticks}'
                else:
                    col_name = col
                if breach_type == 'morethan':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x[-1] > x[:-1]).sum()).fillna(0)
                elif breach_type == 'lessthan':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x[-1] < x[:-1]).sum()).fillna(0)
                elif breach_type == 'mean':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].mean()).sum()).fillna(0).astype(int)
                elif breach_type == 'min':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].min()).sum()).fillna(0).astype(int)
                elif breach_type == 'max':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].max()).sum()).fillna(0).astype(int)
                elif breach_type == 'median':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].median()).sum()).fillna(0).astype(int)
                elif breach_type == '10thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.1)).sum()).fillna(0).astype(int)
                elif breach_type == '25thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.25)).sum()).fillna(0).astype(int)
                elif breach_type == '75thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.75)).sum()).fillna(0).astype(int)
                elif breach_type == '95thquantile':
                    df[col_name] = df[col].rolling(self.last_ticks, min_periods=1).apply(lambda x: (x > x[:].quantile(0.95)).sum()).fillna(0).astype(int)
                else:
                    df[col_name] = (df[col].rolling(self.last_ticks, min_periods=1)
                            .apply(lambda x: (x[-1] > x[:-1]).mean())
                            .astype(int))
        print(f"Shape of dataframe after transform is {df.shape}")
        return df

class PriceDayRangeHourWise(BaseEstimator, TransformerMixin):
    def __init__(self, first_col = 'high',second_col='low',hour_range = [('09:00', '10:30'),('10:30', '11:30')],range_type=['price_range','price_deviation_max_first_col']):
        self.hour_range = hour_range
        self.first_col = first_col
        self.second_col = second_col
        self.range_type = range_type
        
    def fit(self, X, y=None):
        return self    

    def transform(self, df):
        #df = convert_todate_deduplicate(df)
        for r1,r2 in self.hour_range:
            for rt in self.range_type:
                if rt == 'price_range':
                    print(df[self.first_col])
                    s1 = df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max() - df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
                elif rt == 'price_deviation_max_first_col':
                    s1 = df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max()
                elif rt == 'price_deviation_min_first_col':
                    s1 = df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
                elif rt == 'price_deviation_max_second_col':
                    s1 = df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max()
                elif rt == 'price_deviation_min_second_col':
                    s1 = df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).mean() - df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
                else:
                    s1 = df[self.first_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).max() - df[self.second_col].between_time(r1, r2).groupby(pd.Grouper(freq='d')).min()
            s1.index = pd.to_datetime(s1.index) 
            s1 = s1.sort_index()
            c = [int(i) for i in r2.split(':')]
            s1.index = s1.index + pd.DateOffset(minutes=c[0]*60 + c[1])
            s1 = pd.DataFrame(s1,columns=[f"range_{r2.replace(':','')}"])
            df=pd.merge(df,s1, how='outer', left_index=True, right_index=True)
            df[f"range_{r2.replace(':','')}"] = df[f"range_{r2.replace(':','')}"].fillna(method='ffill')
        print(f"Shape of dataframe after transform is {df.shape}")
        return df

class PriceVelocity(BaseEstimator, TransformerMixin):
    def __init__(self, freq='1min',shift=15,shift_column='close'):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.col_name = f'price_velocity_{shift_column}_{freq}_{shift}'
        
    def fit(self, X, y=None):
        return self    # Nothing to do in fit in this scenario
    
    def transform(self, df):
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]         
        df[self.col_name] = df[self.shift_column].subtract(df.shift(self.shift, freq=self.freq)[self.shift_column])
        df[self.col_name] = df[self.col_name]/self.shift
        df[self.col_name] = df[self.col_name].round(3)   
        print(f"Shape of dataframe after transform is {df.shape}") 
        return df

class PricePerIncrement(BaseEstimator, TransformerMixin):
    def __init__(self, freq='1min',shift=15,shift_column='close'):
        self.freq = freq
        self.shift = shift
        self.shift_column = shift_column
        self.col_name = f'price_perincrement_{shift_column}_{freq}_{shift}'
        
    def fit(self, X, y=None):
        return self    # Nothing to do in fit in this scenario
    
    def transform(self, df):
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]         
        df[self.col_name] = df[self.shift_column].subtract(df.shift(self.shift, freq=self.freq)[self.shift_column])
        df[self.col_name] = df[self.col_name].div(df[self.shift_column])*100
        df[self.col_name] = df[self.col_name].round(3)   
        print(f"Shape of dataframe after transform is {df.shape}") 
        return df

