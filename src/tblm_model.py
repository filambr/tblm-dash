import numpy as np
import pandas as pd
import scipy.special as scs
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .utils import *
from .cost_factory import *

def cm_coorection(Y, Y0, kwarg_names, x0, logN, mapp=None):
    if mapp is None:
        m = np.average(np.abs(Y)/np.abs(Y0))
        print(m, 'm')
        temp_kwargs ={}
        temp_kwargs['Cm'] = 10**-6/m
        for x, kw_name in zip(x0, kwarg_names):

            if kw_name == 'CH':
                temp_kwargs[kw_name] = x/m
            elif kw_name == 'r0':
                temp_kwargs[kw_name] = x*np.sqrt(m)
            elif kw_name == 'Rsol':
                temp_kwargs[kw_name] = x*m
            else:
                temp_kwargs[kw_name] = x
        logNN = logN-np.log10(m)
        return Y/m, temp_kwargs, logNN
    else:
        kwargs_list = []
        YY = []
        logNN = []
        for i, (y, y0, x0_sets, kw_names_set) in enumerate(zip(Y, Y0, mapp(x0), mapp(kwarg_names))):
            temp_kwargs = {}
            m = np.average(np.abs(y)/np.abs(y0))
#             print(m)
            temp_kwargs['Cm'] = 10**-6/m
            for x, kw_name in zip(x0_sets, kw_names_set):
                if kw_name == 'CH':
                    temp_kwargs[kw_name] = x/m
                elif kw_name == 'r0':
                    temp_kwargs[kw_name] = x*np.sqrt(m)
                elif kw_name == 'Rsol':
                    temp_kwargs[kw_name] = x*m
                else:
                    temp_kwargs[kw_name] = x
            kwargs_list.append(temp_kwargs)
            logNN.append(logN[i]-np.log10(m))
            YY.append(y/m)
        return YY, kwargs_list, logN



def sum_eq1_F(x0):
    area_0 = np.sum(x0)-1
    return area_0


def cost_get_dist(dist, args):
    # , Cm=1*10**-6,
    #          CH=8*10**-6, r0=2*10**-7, rho=10**4.5, Ydef=50,
    #          Rsol=0.00, d_sub=1.8*10**(-7)):
    f_exp, Y_exp= args['f'], args['Y']
    gener_dist = args['gener_dist']
    regl_matx, logN = args['reg_matx'], args['logN']
    alpha, matx_H = args['alpha'], args['matx_H']
    # print(CH)
    Y_model=np.dot(matx_H, dist)

    mean_logN=np.sum(logN*dist)
    sigma_logN = np.sqrt(np.sum(dist*(logN-mean_logN)**2))
    if np.array(gener_dist==None).any():
        gener_dist=gaussian(logN,mean=mean_logN, std=sigma_logN)

    # Xi = (np.sum(((np.real(1/Y_model) - np.real(1/Y_exp))*np.real(Y_exp)) ** 2)+
    #       np.sum(((np.imag(1/Y_model) - np.imag(1/Y_exp))*np.imag(Y_exp)) ** 2)
    #       )
    Xi_phase = np.sum(((np.angle(Y_model) - np.angle(Y_exp))/np.angle(Y_exp)) ** 2)
    norm = alpha * np.sum((np.dot(regl_matx,dist-gener_dist)**2))
    return (Xi_phase+norm)



class Model:


    def __init__(self):
        self.Factory = {'Single':CostFucntionFactory, 'Multi':MultiCostFucntionFactory}
        self.default_kwargs = pd.DataFrame({"Cm":[1*10**-6],
                              "CH":[8*10**-6],
                              "r0":[2*10**-7],
                              'rho':[10**4.5],
                              'Ydef':[50],
                              'Rsol':[0.00],
                              "d_sub":[1.8*10**(-7)]
                                  })
        self.default_kwargs_status = pd.DataFrame({"Cm":['False'],
                                              "CH":['True'],
                                              "r0":['True'],
                                              'rho':['False'],
                                              'Ydef':['False'],
                                              'Rsol':['True'],
                                              "d_sub":['False']
                                            })

        self.b_min = pd.DataFrame({"Cm":[0.4*10**-6],
                "CH":[7*10**-6],
                "r0":[0.5*10**-7],
                'rho':[10**2.5],
                'Ydef':[1],
                'Rsol':[0.00],
                "d_sub":[0.8*10**(-7)]})
#         print(self.b_min,'self.b_min_init')
        self.b_max = pd.DataFrame({"Cm":[1.5*10**-6],
                "CH":[30*10**-6],
                "r0":[30*10**-7],
                'rho':[10**7],
                'Ydef':[250],
                'Rsol':[250.00],
                "d_sub":[3.8*10**(-7)]})

        self.default_args = pd.DataFrame(dict(
                                f = [np.logspace(-1,5,60)],
                                gener_dist = [None],
                                logN = [N_range()],
                                reg_matx = [regularization_matx()],
                                alpha = [3.31]
                                ))

        Y = np.dot(Model.kernel(self.default_args.to_dict('records')[0]),
                                          gaussian(N_range(), mean=0))
        self.default_args['Y'] = [(1/Y+80)**-1]

    @staticmethod
    def kernel(args, Cm=1*10**-6, CH=8*10**-6,
               r0=2*10**-7, rho=10**4.5, Ydef=50,
               Rsol=0.00, d_sub=1.8*10**(-7)):
        f = args['f']
        logN = args['logN']
        LOGN, F = np.meshgrid(logN, f)
        delta = 1/np.sqrt(np.pi*10**LOGN)*10**-4
        Zdef = 1/Ydef
        CmH = (Cm ** -1 + CH ** -1) ** -1
        k = (d_sub/(rho*np.subtract(CH, CmH)))
        L = delta/r0
        omega = 2*np.pi*F
        lamda = (r0/np.sqrt(2*k))*np.sqrt(omega)
        LAMBDA = (1-1j)*lamda

        LAMBDAL = L*LAMBDA


        H1 = np.multiply(scs.hankel2(1, LAMBDAL), scs.hankel1(1, LAMBDA))
        H2 = np.multiply(scs.hankel1(1, LAMBDAL), scs.hankel2(1, LAMBDA))
        H3 = np.multiply(scs.hankel2(1, LAMBDAL), scs.hankel1(0, LAMBDA))
        H4 = np.multiply(scs.hankel1(1, LAMBDAL), scs.hankel2(0, LAMBDA))

        if np.isnan(H2).any():
            idx = np.where(np.isnan(H2))
            HANF = np.divide(H1-H2, H3-H4)
            HANF2 = np.divide(scs.hankel2(1, LAMBDA),
                              scs.hankel2(0, LAMBDA))
            HANF[idx] = HANF2[idx]
        else:
            HANF = np.divide((H1-H2), (H3-H4))

        CmH = (Cm**-1 + CH**-1)**-1*(delta**2 - r0**2)*np.pi + CH*np.pi*r0**2
        niu = np.multiply(((1 - 1j)*2*np.pi*lamda), HANF)
        Zsub = np.divide((rho/d_sub), niu)
        Zmem = 1/(1j*omega*CmH)
        Ztot = (((Zsub+Zdef)**-1+Zmem**-1))**-1*delta**2*np.pi+Rsol

        return 1/Ztot

    @staticmethod
    def auxilary(args, Cm=1*10**-6,
             CH=8*10**-6, r0=2*10**-7, rho=10**4.5, Ydef=50,
             Rsol=0.00, d_sub=1.8*10**(-7)):
        matx_H = Model.kernel(args={'f':args['f'], 'logN':args['logN']},
                          Cm=Cm, CH=CH, rho=rho, r0=r0,
                          Ydef=Ydef, Rsol=Rsol, d_sub=d_sub)

        arggs = {'f':args['f'],'Y':args['Y'],
                'gener_dist':args['gener_dist'],'reg_matx':args['reg_matx'],
                'alpha':args['alpha'], 'logN':args['logN']}

        arggs.update({'matx_H':matx_H})
        bounds=np.transpose([np.zeros(len(args['logN'])),np.ones(len(args['logN']))])
        cons=[{'type':'eq','fun':sum_eq1_F}]
        solution=minimize(cost_get_dist, x0=np.ones(len(args['logN']))*10**-3,
                          args=arggs, bounds=bounds,
                          constraints=cons,options={'maxiter':500})
        return {'dist': solution.x}


    @staticmethod
    def objective(args, Cm=1*10**-6,
             CH=8*10**-6, r0=2*10**-7, rho=10**4.5, Ydef=50,
             Rsol=0.00, d_sub=1.8*10**(-7)):
        f_exp, Y_exp= args['f'], args['Y']
        gener_dist, dist = args['gener_dist'], args['dist']
        regl_matx, logN = args['reg_matx'], args['logN']
        alpha = args['alpha']
        mean_logN = np.sum(logN*dist)
        sigma_logN = np.sqrt(np.sum(dist*(logN-mean_logN)**2))
        f0 = np.logspace(-2,np.log10(np.min(f_exp)),10)
        f1 = np.logspace(np.log10(np.max(f_exp)),6,10)
        if np.array(gener_dist==None).any():
            gener_dist=gaussian(logN,mean=mean_logN, std=sigma_logN)


        matx_H = Model.kernel(args={'f':args['f'], 'logN':args['logN']},
                            Cm=Cm, CH=CH, rho=rho, r0=r0,
                            Ydef=Ydef, Rsol=Rsol, d_sub=d_sub)
        matx_H0 = Model.kernel(args={'f':f0, 'logN':args['logN']},
                            Cm=Cm, CH=CH, rho=rho, r0=r0,
                            Ydef=Ydef, Rsol=0, d_sub=d_sub)
        matx_H1 = Model.kernel(args={'f':f1, 'logN':args['logN']},
                            Cm=Cm, CH=CH, rho=rho, r0=r0,
                            Ydef=Ydef, Rsol=0, d_sub=d_sub)

        weights = (logN-mean_logN)**2
        Y_model = np.dot(matx_H, dist)
        Y_model0 = np.dot(matx_H0, dist)
        Y_model1 = np.dot(matx_H1, dist)

        # Xi = (np.sum(((np.real(1/Y_model)-np.real(1/Y_exp))*np.real(Y_exp))**2) +
        #       np.sum(((np.imag(1/Y_model)-np.imag(1/Y_exp))*np.imag(Y_exp))**2)
        #       )/2
        Xi_phase = np.sum((np.angle(Y_model) - np.angle(Y_exp)) ** 2)


        norm = alpha * np.sum(((weights)*np.dot(regl_matx,dist-gener_dist))**2)
        features = alpha * (((90-180/np.pi*np.angle(Y_model0))/180/np.pi)**2+((90-180/np.pi*np.angle(Y_model1))/180/np.pi)**2)
        return norm+Xi_phase+np.sum(features)



    def simulate(self, pdf, kwargs=None, args=None):
        if kwargs == None:
            kwargs = self.default_kwargs
        if args == None:
            args == self.default_args
        args['f'] = np.logspace(-2,6,150)
        krnl = self.kernel(args, kwargs)
        return np.dot(krnl, pdf)

    def fit(self, kwargs=None, kwargs_status=None, args=None, b_min=None, b_max=None, algorithm='Powell'):
        from scipy.optimize import differential_evolution
        if kwargs is None:
            kwargs = self.default_kwargs
        if kwargs_status is None:
            kwargs_status = self.default_kwargs_status
        if args is None:
            args = self.default_args
        args_list = Converter(args)
        if b_min is None:
            b_min = self.b_min
        if b_max is None:
            b_max = self.b_max
        
        if len(kwargs.index)>1:
            print('Generating Multi Model')
            CostFactory = self.Factory['Multi']
            objectives = [self.objective for i in kwargs.index]
            CFF = CostFactory(objectives=objectives, auxilary=self.auxilary,
                        kwargs=kwargs, kwargs_status=kwargs_status,
                        b_min=b_min, b_max=b_max)
            bounds = CFF.bounds.reshape(len(CFF.bounds), 2)
            try:
                print('I am about to fit your data')
                if algorithm == 'Differential evolution':
                    self.solution = differential_evolution(CFF.cost, args=(args_list,),
                                                            bounds=bounds, popsize=15,
                                                            maxiter=50, polish=True,
                                                            updating='deferred', tol=1e-2,
                                                           strategy='randtobest1bin')
                elif algorithm == 'Powell':
                    print(bounds, 'Powell bounds')
                    self.solution = minimize(CFF.cost, x0=(bounds[:,0]+bounds[:,1])/2 ,args = (args_list,), bounds=bounds, method='Powell')
                print('fewh! That was a lot of work!')
            except Exception as e:
                print(e)
            for name in kwargs_status:
                if 'Share' in np.array(kwargs_status[name]):
                    kwargs_status[name] = ['False' for x in kwargs_status[name]]
                    kwargs_status[name][0] = 'Share'
            fitted_pars = self.solution.x
            fitted_par_names = np.concatenate(
                np.array([fetch_params_to_fit(series) for series
                in kwargs_status.to_dict('records')], dtype='object'
                )
            )

            fitted_kwargs = [dict(zip(fp_names,fp)) for fp, fp_names in
                             zip(CFF.mapp(fitted_pars),CFF.mapp(fitted_par_names))
            ]

            kwargs = kwargs.to_dict('records')

            kwargs_final = [kw.update(fkw) for kw, fkw in zip(kwargs, fitted_kwargs)]
            kernel = [Model.kernel(argss, **fitted_kwargss) for
                      argss, fitted_kwargss in zip(args_list, kwargs)
            ]

            pdf = Model.auxilary(args_list[0], **kwargs[0])
            Y = [np.dot(kernell, pdf['dist']) for kernell in kernel]
            Y, fitted_kwargs, args['logN'] = cm_coorection(Y,args['Y'], fitted_par_names,
                                                           fitted_pars, args['logN'],
                                                           mapp=CFF.mapp
            )
            for f, Yfit, Yexp in zip(args['f'], Y, args['Y']):
                plt.plot(np.log10(f), 180/np.pi*np.angle(Yfit))
                plt.scatter(np.log10(f), 180/np.pi*np.angle(Yexp))
            plt.show()
            for f, Yfit, Yexp in zip(args['f'], Y, args['Y']):
                plt.scatter(np.real(1/(1j*np.pi*2*f*(1/Yexp))), -np.imag(1/(1j*np.pi*2*f*(1/Yexp))))
                plt.plot(np.real(1/(1j*np.pi*2*f*(1/Yfit))), -np.imag(1/(1j*np.pi*2*f*(1/Yfit))))
            plt.show()
            plt.plot(args['logN'][0], pdf['dist'])
            plt.show()
            self.Data_fitted = []
            for fexp, yfit in zip(args['f'], Y):
                data_fitted = pd.DataFrame({'f':fexp,
                              'zr':np.real(1/yfit),
                              'zi':np.imag(1/yfit),
                              })
                self.Data_fitted.append(eis_df(data_fitted))
            self.pdf = {'dist':pdf['dist'], 'logN':args['logN'][0]}
            self.fitted_kwargs = fitted_kwargs
        else:
            print('Generating Single Model')
            CostFactory = self.Factory['Single']
            objective = Model.objective
            CFF = CostFucntionFactory(objective=objective, auxilary=self.auxilary,
                                      kwargs=kwargs, kwargs_status=kwargs_status,
                                      b_min=b_min, b_max=b_max)
#             print(self.b_min,'self.b_min')
#             print(b_min)
#             print(self.b_max,'self.b_min')
#             print(b_max)
            bounds = CFF.bounds.reshape(len(CFF.bounds), 2)
            '''plt.plot(np.log10(self.default_args['f']), np.angle(self.default_args['Y']))
            plt.show()
            plt.plot(np.log10(args['f']), np.angle(args['Y']))
            plt.show()'''

            try:
                print('I am about to fit your data')
                if algorithm == 'Differential evolution':
                    self.solution = differential_evolution(CFF.cost, args=(args_list,),
                                                            bounds=bounds, popsize=15,
                                                            maxiter=50, polish=True,
                                                            updating='deferred', tol=1e-2,
                                                           strategy='randtobest1bin')
                elif algorithm == 'Powell':
#                     print(bounds, 'Powell bounds')
                    self.solution = minimize(CFF.cost, x0=(bounds[:,0]+bounds[:,1])/2 ,args = (args_list,), bounds=bounds, method='Powell')

                print('fewh! That was a lot of work!')
            except Exception as e:
                print(e)
            fitted_pars = self.solution.x
            fitted_par_names = fetch_params_to_fit(kwargs_status)
            fitted_kwargs = dict(zip(fitted_par_names,fitted_pars))

            kwarg_finals = kwargs.to_dict('records')[0]
            kwarg_finals.update(fitted_kwargs)
            pdf = Model.auxilary(args_list, **kwarg_finals)
            kernel = Model.kernel(args_list,  **kwarg_finals)
            Y = np.dot(kernel, pdf['dist'])
            Y, fitted_kwargs, args_list['logN'] = cm_coorection(Y,args_list['Y'],
                                                                  fitted_par_names,
                                                                  fitted_pars, args_list['logN']
            )
            self.fitted_kwargs = [fitted_kwargs]
            plt.scatter(np.log10(args_list['f']), 180/np.pi*np.angle(args_list['Y']))
            plt.plot(np.log10(args_list['f']), 180/np.pi*np.angle(Y))
            plt.show()
            plt.scatter(np.real(1/(1j*np.pi*2*args_list['f']*args_list['Y']**-1)), -np.imag(1/(1j*np.pi*2*args_list['f']*args_list['Y']**-1)))
            plt.plot(np.real(1/(1j*np.pi*2*args_list['f']*(1/Y))), -np.imag(1/(1j*np.pi*2*args_list['f']*(1/Y))))
            plt.show()
            plt.plot(args_list['logN'], pdf['dist'])
            plt.show()

            data_fitted = pd.DataFrame({'f':args_list['f'],
                           'zr':np.real(1/Y),
                           'zi':np.imag(1/Y),
                           })

            self.Data_fitted = [eis_df(data_fitted)]

            self.pdf = {'dist':pdf['dist'], 'logN':args_list['logN']}
        return [d for d in self.Data_fitted], [self.pdf], self.fitted_kwargs
