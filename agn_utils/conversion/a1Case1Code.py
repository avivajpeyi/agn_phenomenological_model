def mysqrt(x): return np.sqrt((1.+0j)*x)


def a1case1():
    aux0=chieff*(q*((1.+q)*(Q*((np.cos(Δϕ))*((np.cos(θ2))*((np.sin(θ2))*(Tan[θ1])))))));
    aux1=(chip**2)*(q*(Q*((np.cos(Δϕ))*((np.cos(θ2))*((np.sin(θ2))*(Tan[θ1]))))));
    aux2=(chip**2)-((chieff**2)*((((1.+q)**2))*((((np.sin(Δϕ))**2))*(((Tan[θ1])**2)))));
    aux3=((chip**2)*((q**2)*((((np.cos(θ2))**2))*(((Tan[θ1])**2)))))+((Q**2)*((((np.sin(θ2))**2))*aux2));
    aux4=(np.cos(θ2))*((((Sec[θ1])**2))*(mysqrt((((np.cos(θ1))**4.)*((-2.*aux1)+aux3)))));
    aux5=(-(Sec[θ1])*((aux0+(q*aux4))-(chieff*((1.+q)*((Q**2)*(((np.sin(θ2))**2)))))));
    aux6=(-2.*(q*(Q*((np.cos(Δϕ))*((np.cos(θ2))*((np.sin(θ2))*(Tan[θ1])))))))+((q**2)*((((np.cos(θ2))**2))*(((Tan[θ1])**2))));
    output=aux5/(((Q**2)*(((np.sin(θ2))**2)))+aux6);
    return output
