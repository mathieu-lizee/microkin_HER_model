import numpy as np
from scipy.integrate import quad
from scipy import constants as const
import matplotlib.pyplot as plt
from numpy import*
from matplotlib.pyplot import*
from scipy.optimize import fsolve
from scipy.stats import linregress
from isocurrent_lines import *
eV=1.6e-19*const.N_A#J/mol
F=96485.3329


def eta_chem(solver_params):
    T=solver_params['T'];E_H_ads=solver_params['E_H_ads'];theta=solver_params['theta']    
    try:intermediate_entropy=solver_params['isotherm']
    except:intermediate_entropy=0 #np.log(theta/(1-theta))*const.R
    return E_H_ads-intermediate_entropy*T

def Volmer(solver_params):
    eta=solver_params['eta'];T=solver_params['T'];Ea1=solver_params['Ea1'];k1_0=solver_params['k1_0']*solver_params['c_surface'];alpha=solver_params['alpha'];theta=solver_params['theta']
    etaN = eta_chem(solver_params)
    j1=k1_0*np.exp(-(Ea1 - alpha*F*(eta - etaN))/(const.R*T))*(1-theta)
    jm1=j1*np.exp(-(F*(eta-etaN))/(const.R*T))*theta/(1-theta)
    return j1,jm1


def Tafel_rate(solver_params,etaN):
    T=solver_params['T'];Ea2=solver_params['Ea2'];k2_0=solver_params['k2_0'];alpha=solver_params['alpha'];theta=solver_params['theta']
    try:E_H2=solver_params['E_H2']
    except:E_H2=0
    etaN = eta_chem(solver_params)
    j2=2*k2_0*np.exp(-(Ea2 - 2*alpha*F*etaN)/(const.R*T))*theta**2
    jm2= j2/theta**2*(1-theta)**2*np.exp((E_H2-2*etaN)*F/(const.R*T))
    return j2,jm2

def Heyrovsky_rate(solver_params,etaN):
    eta=solver_params['eta'];T=solver_params['T'];Ea2=solver_params['Ea2'];k2_0=solver_params['k2_0'];alpha=solver_params['alpha'];theta=solver_params['theta']
    try:E_H2=solver_params['E_H2']
    except:E_H2=0
    etaN = eta_chem(solver_params)
    j2= k2_0*np.exp(-(Ea2 - alpha*F*(eta+etaN))/(const.R*T))*theta
    jm2= j2*(1-theta)/theta*np.exp((E_H2-eta-etaN)*F/(const.R*T))
    return j2,jm2



def her_current(solver_params):
    steps=solver_params['N_steps']
    if(steps==2):    
        def balance_theta(theta):
            solver_params['theta']=theta
            etaN = eta_chem(solver_params)
            T=solver_params['T']
            if(solver_params['step1']=='BV'):
                j1,jm1 = Volmer(solver_params)
                j=j1-jm1
            else:raise Exception('Step1 must be BV')

            if(solver_params['step2']=='Heyrovsky'):j2,jm2 = Heyrovsky_rate(solver_params,etaN)
            elif(solver_params['step2']=='Tafel'):j2,jm2 = Tafel_rate(solver_params,etaN)
            else:raise Exception('Step2 must be Tafel or Heyrovsky')
            return (j-j2+jm2)**2
        
        def HER_2steps_c0(c_surface):
            solver_params['c_surface']=c_surface
            x,solution,ier,mesg = fsolve(balance_theta, [0.1],full_output=True)
            theta = x[0];solver_params['theta']=theta;etaN = eta_chem(solver_params)
            if(solver_params['step1']=='BV'):j1,jm1 = Volmer(solver_params)
            else:raise Exception('Step1 must be BV')
            if(solver_params['step2']=='Tafel'):j2,jm2 = Tafel_rate(solver_params,etaN)
            elif(solver_params['step2']=='Heyrovsky'):j2,jm2 = Heyrovsky_rate(solver_params,etaN)
            else:raise Exception('Step2 must be Tafel or Heyrovsky')
            j=j1-jm1
            return j, theta, etaN,jm1
        
    elif(steps==1):
        def BV_1step(c_surface):
            solver_params['c_surface']=c_surface
            if(solver_params['step1']=='BV'):
                eta=solver_params['eta'];T=solver_params['T'];Ea1=solver_params['Ea1'];k1_0=solver_params['k1_0']*c_surface;alpha=solver_params['alpha']
                j1=k1_0*exp(-(Ea1 - alpha*F*eta)/(const.R*T))
                jm1=j1*np.exp(-(F*(eta))/(const.R*T))
                j=j1-jm1
            else:raise Exception('Step1 must be BV')
            return j,jm1
    else:
        raise ValueError('N_steps must be 1 or 2')


    if(solver_params['mass_transport']):

        def balance_c0(c_surface,steps=steps,setup=solver_params['setup']):
            if(steps==2):j, theta, etaN,jm1 = HER_2steps_c0(c_surface)
            elif(steps==1):j,jm1 = BV_1step(c_surface)
            if(setup=='RDE'):c_j=c_drop_RDE(j,solver_params)
            elif(setup=='UME'):c_j=c_drop_UME(j,solver_params)
            else: raise ValueError('setup must be UME or RDE')
            return c_j-c_surface

        x,solution,ier,mesg = fsolve(balance_c0, solver_params['c0'],full_output=True)
        c_surface = x[0];solver_params['c_surface']=c_surface
        if(steps==2):j, theta, etaN,jm1 = HER_2steps_c0(c_surface)
        elif(steps==1):j,jm1 = BV_1step(c_surface);theta=0

        ##WARNING currents in A/cm^2
        j*=1e3#to mA/cm^2
        jm1*=1e3
    else:
        c_surface=solver_params['c0'];solver_params['c_surface']=c_surface
        if(steps==2):j, theta, etaN,jm1 = HER_2steps_c0(c_surface)
        elif(steps==1):j,jm1 = BV_1step(c_surface);theta=0
    return j, c_surface, theta,jm1


def Coverage(params):
    theta = her_current(params)[2]
    return theta
def Arrhenius(params):
    delta_T = 0.2;T=params['T']  # Small temperature change for numerical differentiation
    j_plus = her_current({**params, 'T': T + delta_T})[0]
    j_minus = her_current({**params, 'T': T - delta_T})[0]
    apparent_Ea, pre_exponential_factor, r_value, p_value, std_err = linregress(1/array([T - delta_T, T + delta_T]), log(array([j_minus, j_plus])))
    apparent_Ea = -apparent_Ea * 8.314  # Convert slope to activation energy in J/mol
    return(apparent_Ea, pre_exponential_factor/log(10))
def Tafel(params):
    delta_eta = 1e-3  # Small overpotential change for numerical differentiation
    j_plus_eta = her_current({**params, 'eta': params['eta'] + delta_eta})[0]
    j_minus_eta = her_current({**params, 'eta': params['eta'] - delta_eta})[0]
    tafel_slope = 1e3 * (2 * delta_eta) / (np.log10(j_plus_eta) - np.log10(j_minus_eta))
    return tafel_slope#mV/dec
def Conway(params):
    dT=3
    tafel_slopes_conway = [Tafel({**params, 'T': params['T'] + t}) for t in [-dT, 0, dT]]
    slope, intercept, r_value, p_value, std_err = linregress([params['T'] - dT, params['T'], params['T'] + dT], tafel_slopes_conway)
    alpha_H = intercept
    alpha_S = slope
    return alpha_H, alpha_S
def reaction_order(params):
    delta_c=1e-2
    j_plus = her_current({**params, 'c0': params['c0'] * (1 + delta_c)})[0]
    j_minus = her_current({**params, 'c0': params['c0'] * (1 - delta_c)})[0]
    order = (log(j_plus) - log(j_minus)) / (log(params['c0'] * (1 + delta_c)) - log(params['c0'] * (1 - delta_c)))
    return order
def degrees_of_rate_control(params, k1_0=1):
    delta_k1_0 = 1e-2;k1_0=params['k1_0']
    j_k1_plus = her_current({**params, 'k1_0': k1_0 * (1 + delta_k1_0)})[0]
    j_k1_minus = her_current({**params, 'k1_0': k1_0 * (1 - delta_k1_0)})[0]
    degree_k1_0 = (log(j_k1_plus) - log(j_k1_minus)) / (log(k1_0 * (1 + delta_k1_0)) - log(k1_0 * (1 - delta_k1_0)))

    if(params['N_steps'])==2:
        delta_k2 = 1e-2
        j_k2_plus = her_current({**params, 'k2_0': params['k2_0'] * (1 + delta_k2)})[0]
        j_k2_minus = her_current({**params, 'k2_0': params['k2_0'] * (1 - delta_k2)})[0]
        degree_k2 = (log(j_k2_plus) - log(j_k2_minus)) / (log(params['k2_0'] * (1 + delta_k2)) - log(params['k2_0'] * (1 - delta_k2)))
    else:degree_k2=0

    if(params['N_steps'])==2:
        delta_binding=1e-2
        j_binding_plus = her_current({**params, 'E_H_ads': params['E_H_ads'] * (1 + delta_binding)})[0]
        j_binding_minus = her_current({**params, 'E_H_ads': params['E_H_ads'] * (1 - delta_binding)})[0]
        degree_binding = -(log(j_binding_plus) - log(j_binding_minus)) / log((1 + delta_binding)/(1 - delta_binding))
    else:degree_binding=0

    delta_Ea1=1e-2
    j_Ea1_plus = her_current({**params, 'Ea1': params['Ea1'] * (1 + delta_Ea1)})[0]
    j_Ea1_minus = her_current({**params, 'Ea1': params['Ea1'] * (1 - delta_Ea1)})[0]
    degree_Ea1 = -(log(j_Ea1_plus) - log(j_Ea1_minus)) / log((1 + delta_Ea1)/(1 - delta_Ea1))
    
    if(params['N_steps'])==2:
        delta_Ea2=1e-2
        j_Ea2_plus = her_current({**params, 'Ea2': params['Ea2'] * (1 + delta_Ea2)})[0]
        j_Ea2_minus = her_current({**params, 'Ea2': params['Ea2'] * (1 - delta_Ea2)})[0]
        degree_Ea2 = -(log(j_Ea2_plus) - log(j_Ea2_minus)) / log((1 + delta_Ea2)/(1 - delta_Ea2))
    else:degree_Ea2=0

    return(degree_k1_0, 0, degree_k2, 0, degree_binding)
    #return(degree_Ea1,0,degree_Ea2,0,degree_binding)# , delta_Ea1, )


def compute_kinetic_parameters(params):
    coverage = Coverage(params)
    apparent_Ea, pre_exponential_factor = Arrhenius(params)
    tafel_slope = Tafel(params)
    (alpha_H, alpha_S) = Conway(params)
    degree_k1_0, degree_km1_0, degree_k2, degree_km2, degree_binding = degrees_of_rate_control(params)
    return {
        "Coverage (θ)": coverage,
        "Apparent Activation Energy (Ea)": apparent_Ea,
        "Pre-exponential Factor (A)": pre_exponential_factor,
        "Tafel Slope (mV/dec)": tafel_slope,
        "Alpha_H": alpha_H,
        "Alpha_S": alpha_S,
        "Degree of Rate Control (k1_0)": degree_k1_0,
        "Degree of Rate Control (km1_0)": degree_km1_0,
        "Degree of Rate Control (k2)": degree_k2,
        "Degree of Rate Control (km2_0)": degree_km2,
        '"Degree of Rate Control (Binding Energy)': degree_binding,
        "Sum": degree_k1_0 + degree_km1_0 + degree_k2 + degree_km2,
        "Order": reaction_order(params)}

def plot_compensation(ax,params,etas=linspace(0,0.3,100),sweep_tab=[1],param_sweep='None',
                      scatter_points=True,show_tafel=False,ax1=None,ax2=None):
    etas_arg=etas.copy()
    cmap=cm.inferno;norm=plt.Normalize(vmin=min(etas),vmax=1*max(etas))
    cmap2=cm.RdYlBu_r;norm2=plt.Normalize(vmin=0,vmax=len(sweep_tab)-1)
    max_EA_app=[]
    for k,x in enumerate(sweep_tab):
        if(param_sweep!='None'):params={**params,param_sweep:x}
        else:etas_restrict=etas_arg;eta_min,eta_max=min(etas_restrict),max(etas_restrict)
        Ea_app, A_app=array([Arrhenius({**params, 'eta': eta}) for eta in etas_restrict]).transpose()#A_app=10**A_app
        if(scatter_points):ax.scatter(Ea_app/1000,A_app,c=array([cmap(norm(float(x))) for x in etas]),s=20,alpha=0.5)
        ax.plot(Ea_app/1000,A_app,color=cmap2(norm2(k)),label=f'{param_sweep}={x:.1e}')
        max_EA_app.append(etas_restrict[argmax(Ea_app)])
        if(show_tafel):
            plot_tafel(ax1,ax2,params, etas=np.linspace(eta_min,eta_max,30),plot_=False,temps=array([params['T']]),color1=cmap2(norm2(k)))
    ax.set_xlabel("Apparent Ea [kJ/mol]"); ax.set_ylabel(f"log A ({r'mA/cm$^2$'})")#;ax.set_title('Kinetic map')
    plotIsocurrentLines(ax,range(-20,20),isocurrent_temperature=30,label_units='')
    if(scatter_points):
        cbar=colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='η [V]',ax=ax,shrink=0.4)#,anchor=(-2,0))
    return(etas,Ea_app,A_app,max_EA_app[0])

# === Plot polarization curves ===
def plot_polarization(ax,ax2,ax3,params, temps=np.linspace(280,340,4),
                      etas=np.linspace(0.01,0.3,20),aH=1):
    norm=plt.Normalize(vmin=min(temps),vmax=max(temps));cmap=cm.RdBu
    for T in temps:
        data = [her_current({**params, 'eta': e, 'T': T}) for e in etas]
        currents, c_tab,thetas,jm1 = np.array(data).T
        ax.plot(etas, currents, label=f"T={round(T,0)} K",color=cmap(norm(int(T))))
        #ax_=ax.twinx()
        if(params['N_steps']==2):
            ax.plot(etas, jm1,color=cmap(norm(int(T))),linestyle='--')
            ax2.plot(etas, thetas, ls='-', color=ax.get_lines()[-1].get_color())
            ax2.set_xlabel("Overpotential η [V]");ax2.set_ylabel("Surface coverage θ")
        ax3.plot(etas, c_tab, ls='-', color=ax.get_lines()[-1].get_color())
        ax3.set_xlabel("Overpotential η [V]");ax3.set_ylabel("Concentration (M)")
    ax.set_xlabel("Overpotential η [V]"); ax.set_ylabel(f"Current density ({r'mA/cm$^2$'})")
    ax.set_yscale('log')
    subplots_adjust(wspace=0.45)
    ax.legend(loc='upper left', fontsize=8)
    return(etas,currents,thetas)


def plot_arrhenius(ax,params, temps=np.linspace(280,340,10), etas=np.linspace(0.01,0.3,5)):
    norm=plt.Normalize(vmin=min(etas),vmax=1.1*max(etas));cmap=plt.get_cmap('inferno')
    for e in etas:
        data = [her_current({**params, 'eta': e, 'T': T}) for T in temps]
        currents, thetas,eta_chems,jm1 = np.array(data).T
        ax.plot(1e3/temps, log(currents), label=fr"$\eta$={round(e,1)} V",color=cmap(norm(float(e))))
        slope, intercept, r_value, p_value, std_err = linregress(1e3 / temps, np.log(currents))
        ax.plot(1e3 / temps, slope * (1e3 / temps) + intercept, '--', color=ax.get_lines()[-1].get_color())
        ax.text(1e3 / temps[-1], np.log(currents[-1]), f"$R^2$={r_value**2:.3f}", fontsize=8, color=ax.get_lines()[-1].get_color())
    ax.set_ylabel((f"log j ({r'mA/cm$^2$'})")); ax.set_xlabel("1e3/T")

# === Arrhenius analysis ===
def arrhenius_analysis(params,etas=np.linspace(0.0,0.3,5),temps=np.linspace(280,340,7)):
    prefactors, act_energies,theta_grid = {}, {}, {}
    for eta in etas:
        thetas= np.array([her_current({**params, 'eta': eta, 'T': T})[1] for T in temps])
        Ea_app, A_app=Arrhenius({**params, 'eta': eta, 'T': 300});A_app=10**A_app
        act_energies[eta], prefactors[eta] = Ea_app, A_app
        theta_grid[eta]=thetas
    return act_energies, prefactors, theta_grid 

def plot_DRC(ax,params, etas=np.linspace(0.0,0.3,5)):
    DRC1, DRCm1,DRC2,DRCm2 = {}, {}, {},{}
    for eta in etas:
        degree_k1_0, degree_km1_0, degree_k2, degree_km2,drc_binding = degrees_of_rate_control({**params, 'eta': eta})
        DRC1[eta], DRCm1[eta], DRC2[eta], DRCm2[eta] = degree_k1_0, degree_km1_0, degree_k2, degree_km2
    ax.plot(etas, [DRC1[eta] for eta in etas], label=r"$X_{1}$",color='r')
    ax.plot(etas, [DRC2[eta] for eta in etas], label=r"$X_{2}$",color='steelblue')
    ax.set_xlabel("Overpotential η [V]");ax.set_ylabel(r"Degree of Rate Control $X_i$")
    ax.legend(fontsize=8,ncol=1)

def plot_tafel(ax,ax2,params, temps=np.linspace(280,340,4), etas=np.linspace(0.01,0.3,20),plot_=False,
               color1=None):
    norm=plt.Normalize(vmin=min(temps),vmax=max(temps));cmap=cm.RdBu
    tafel_slopes = {T: {} for T in temps}
    for T in temps:
        if(color1==None):color_T=cmap(norm(float(T)))
        else:color_T=color1
        data = array([her_current({**params, 'eta': e, 'T': T}) for e in etas])
        for i in range(1,len(etas)-1):
            slope=Tafel({**params, 'T': T, 'eta': etas[i]});tafel_slopes[T][etas[i]] = slope;origin=0
        currents, thetas,eta_chems,jm1 = np.array(data).T
        ax.plot(log10(currents), etas, label=f"T={T} K",color=color_T)
        ax2.plot(etas[1:-1],tafel_slopes[T].values(), ls='-', color=ax.get_lines()[-1].get_color())
    ax.set_xlabel("log j (A/cm$^2$)"); ax.set_ylabel(r"$\eta$ (V)")
    ax2.set_xlabel(r"$\eta$ (V)");ax2.set_ylabel("Tafel slope [mV/decade]")

def plot_analysis(params,eta_min=1e-2,
                  eta_max=0.3,N_eta=5,temps=np.linspace(280,340,4),titre=''):
    fig=figure(figsize=(14,6))
    if(params['N_steps'])==2:ax1=subplot(251);ax2=subplot(252);ax3=subplot(253);ax6=subplot(254);ax8=subplot(255)
    else:ax1=subplot(241);ax2=subplot(242);ax3=subplot(242);ax6=subplot(243);ax8=subplot(244)
    ax4=subplot(247);ax7=subplot(248);ax5=subplot(223)
    (eta_tab,current_tab,theta_tab)=plot_polarization(ax1,ax3,ax2,params, etas=np.linspace(eta_min,eta_max,N_eta),temps=temps)
    plot_tafel(ax6,ax8,params, etas=np.linspace(eta_min,eta_max,N_eta),plot_=False,temps=temps)
    plot_arrhenius(ax4,params, etas=np.linspace(eta_min,eta_max,5),temps=temps)
    (eta_tab,Ea_tab,A_tab,eta_turnover)=plot_compensation(ax5,params, etas=np.linspace(eta_min,eta_max,2*N_eta))
    plot_DRC(ax7,params,etas=np.linspace(eta_min,eta_max,N_eta))
    subplots_adjust(wspace=0.6,hspace=0.4)
    savefig(f'Figures/Fig_mass_transport_model'+titre+'.pdf',bbox_inches='tight')
    show()
    return(eta_tab,current_tab,theta_tab,Ea_tab,A_tab)


#Mass transport
#UME
def c_disk(j,R=1e-6,c_0=1,D=1e-9,T=300,c_gly=0,n_electrons=2):
    D_T=D*(mu(0,300-273.15)/mu(c_gly,T-273.15))**0.9
    c_0*=1e3;j_0=j*1e4;F=96485.3329#j_0 in s^-1/m^2, c_0 in mol/m^3
    Dc_surf=j_0*np.pi*R/(4*F*D_T)/n_electrons
    return (c_0-Dc_surf)*1e-3#back to M

#RDE
def m_O_RDE_levitch(omega_RPM,D=1e-9,c_gly=0,T=300):
    mu_=mu(c_gly,T-273.15);rho=(1+0.26*c_gly)*1e3;nu=mu_/rho
    omega_RPM=max(omega_RPM,1e-2)
    omega=omega_RPM*2*np.pi/60
    m_O_RDE=(0.620)*D**(2/3)*omega**(1/2)*nu**(-1/6)
    return(m_O_RDE)

def c_drop_UME(j,solver_params):
    R=solver_params['R'];c_0=solver_params['c0'];T=solver_params['T']
    try:c_gly=solver_params['c_gly']
    except:c_gly=0
    try:D_reactant=solver_params['D_reactant']
    except:D_reactant=1e-9
    D_T=D_reactant*(mu(0,300-273.15)/mu(c_gly,T-273.15))**0.9
    c_0*=1e3;j_0=j*1e4;F=96485.3329#j_0 in A/m^2, c_0 in mol/m^3
    Dc_surf=j_0*np.pi*R/(4*F*D_T)
    return (c_0-Dc_surf)*1e-3#back to M

def c_drop_RDE(j,solver_params):
    omega_RPM=solver_params['omega_RPM'];c_0=solver_params['c0'];T=solver_params['T']
    try:c_gly=solver_params['c_gly']
    except:c_gly=0
    try:beta=solver_params['beta']
    except:beta=1
    try:D_reactant=solver_params['D_reactant']
    except:D_reactant=1e-9
    D_T=D_reactant*(mu(0,300-273.15)/mu(c_gly,T-273.15))**beta
    m_O_RDE=m_O_RDE_levitch(omega_RPM=omega_RPM,D=D_T,c_gly=c_gly,T=T)
    c_0*=1e3;j_0=j*1e4;F=96485.3329#j_0 in A/m^2, c_0 in mol/m^3
    return(c_0-j_0/F/m_O_RDE)*1e-3#back to M


#Viscosity of water-glycerol mictures
def mu_w(T):return(1.79*e**((-1230-T)*T/(36100+360*T)))
def mu_g(T):return(12100*e**((-1233+T)*T/(9900+70*T)))
def a_(T):return(0.705-0.0017*T)
def b_(T):return((4.9+0.036*T)*a_(T)**2.5)
def alpha_(Cm,T):
    (a,b)=(a_(T),b_(T))
    return(1-Cm+(a*b*Cm*(1-Cm))/(a*Cm+b*(1-Cm)))
def mu(gly_content,T):
    alpha=alpha_(gly_content,T)
    return(mu_g(T)**(1-alpha)*mu_w(T)**alpha*1e-3)