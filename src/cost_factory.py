import numpy as np
import pandas as pd
from numpy import vectorize
from .utils import *

@vectorize
def trelien(tring):
    df = pd.DataFrame({'False':[0], 'True':[1], 'Share':[2]})
    return df[tring]





class CostFucntionFactory:
    def __init__(self, objective, auxilary,
                 kwargs: pd.DataFrame, kwargs_status: pd.DataFrame, b_min, b_max):
        self.kwargs = kwargs
        self.kwargs_status = kwargs_status
        self.objective = self.format_function(objective, self.kwargs, self.kwargs_status)
        self.auxilary = self.format_function(auxilary, self.kwargs, self.kwargs_status)
        self.cost = self.cost_generator_with_shared_distirbution(self.objective, self.auxilary)
        self.bounds = self.get_bounds(b_min, b_max, self.kwargs_status)
    def format_function(self, cost_function, kwargs, kwargs_status):
        '''
        transform function of the form  fn(args, **kwargs) to the form fn(x0, args, kwargs=kwargs),
          x0 - array of parameters that will be fitted
          **kwargs - set to default values which are input kwargs,
                except for those that will be fitted.
          args - dictionary of args needed for evaluation of the function

        '''
        try:
            kwargs_status = Converter(kwargs_status)
            params_to_fit = fetch_params_to_fit(kwargs_status)
            kwargs = Converter(kwargs)
            def out_fn(x0, args, kwargs_to_fit=params_to_fit, kwargs=kwargs):

                par_dict = dict(zip(kwargs_to_fit, x0))
                kwargs.update(par_dict)
                Output = cost_function(args, **kwargs)
                return Output
            return out_fn
        except Exception as tye:
            print(tye, 'in format function')

    def cost_generator_with_shared_distirbution(self, fn, auxilary):
        '''
        Compiles single objective function of the form fn(x0, args)
        '''
        print('generating "single" cost function')
        def cost(x0, args):
            new_arg = auxilary(x0, args)
            args.update(new_arg)
            msrd = fn(x0, args)
            return msrd

        return cost

    def get_bounds(self, b_min, b_max, kwargs_status):
        """
        loops over rows in df, returns column names where cell value != False
        then transforms into array to corresond to names in x0
        """
        params_to_fit = fetch_params_to_fit(kwargs_status)
        bounds = np.transpose([b_min[params_to_fit], b_max[params_to_fit]])
        return bounds


class MultiCostFucntionFactory:
    def __init__(self, objectives, auxilary,
                 kwargs: pd.DataFrame, kwargs_status: pd.DataFrame, b_min, b_max):
        self.objectives = self.format_function(objectives, kwargs, kwargs_status)
        self.auxilary = self.format_function([auxilary], kwargs, kwargs_status)[0]
        self.kwargs = kwargs
        self.kwargs_status = kwargs_status
        self.mapp = self.map_generator(kwargs_status)
        self.cost = self.cost_generator_with_shared_distirbution(self.objectives, self.auxilary, self.mapp)
        self.bounds = self.get_bounds(b_min, b_max, kwargs_status)

    def format_function(self, cost_function, kwargs, kwargs_status):
      '''
      transform function of the form  fn(**kwargs) to the form fn(x0: list, args),
      where x0 is list of relevent parameters
      '''
      formated_functions = []
      for cst, kw, kwst in zip(cost_function, kwargs.to_dict('records'), kwargs_status.to_dict('records')):
          params_to_fit = fetch_params_to_fit(kwst)

          def out_fn(x0, args, kwargs_to_fit=params_to_fit, kwargs=kw):
              par_dict = dict(zip(kwargs_to_fit, x0))
              kwargs.update(par_dict)

              Output = cst(args, **kwargs)
              return Output
          formated_functions.append(out_fn)
      return formated_functions


    def cost_generator_with_shared_distirbution(self, fn_list: list, auxilary, mapp):
        '''
        Compiles single objective function of the form fn(x0, args)
        '''
        def cost(x0, args):
            # auxilary calculates additional args that are sheared between functions in fn_list
            x0 = mapp(x0)
            new_arg = auxilary(x0[0], args[0])

            msrd = []
            for x, fn, arg in zip(x0, fn_list, args):
                arg.update(new_arg)
                msrd.append(fn(x, arg))
            return np.average(msrd)
        return cost

    def get_bounds(self, b_min, b_max, kwargs_status):
        """
        loops over rows in df, returns column names where cell value != False
        then transforms into array to corresond to names in x0
        """
        # coppy of kwargs_status is needed in order not to modify input veriable after function execution
        kwargs_status_ = pd.DataFrame()
        for name in kwargs_status:
            kwargs_status_[name] = kwargs_status[name]
            if np.isin('Share', kwargs_status[name].tolist()):
                kwargs_status_[name] = ['False' for value in kwargs_status[name]]
                kwargs_status_[name][0] = 'Share'
        fit_kwargs = np.hstack([row[row !='False'].index.tolist()
                                for row in kwargs_status_.iloc
                              ])
        return np.transpose([b_min[fit_kwargs], b_max[fit_kwargs]]).reshape(len(fit_kwargs),2)


    def map_generator(self, kwargs_status =  pd.DataFrame(data = {'kwargs0':[True,False,True,True],
                                                            'kwargs1': ['Share','Share','Share','Share'],
                                                            'kwargs2': [False,False,True,True]})):

        '''
        generated function of the form map(np.array(x0)) -> list[x0, x1, x2 ...], where each x0 is set of parameters for a function
        '''
        arr = trelien(kwargs_status.to_numpy())
        arr_coppy = arr*1

        idx_two = np.where(arr[0] == 2)
        idx_ones = np.where(arr[1:] == 1)

        arr_coppy[0] = (np.cumsum(arr[0].astype(bool)))*arr[0].astype(bool)
        idx0 = (np.cumsum(arr[0].astype(bool)))*arr[0].astype(bool)
        arr_coppy[1:][idx_ones] = np.cumsum(arr_coppy[1:][idx_ones])+np.max(idx0)
        arr_coppy[1:,idx_two] = idx0[idx_two]
        arr_coppy[0] = idx0

        idx =[]
        for row in arr_coppy:
            idx.append(list(row[row != 0]-1))

        def map(x0):
            if type(x0) == list:
                x0 = np.array(x0)
                print ("TypeError: x0 must be numpy array, not list, converting it to numpy array")

            return [x0[id] for id in idx]
        return map
