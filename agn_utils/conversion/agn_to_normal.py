import numpy as np
from numpy import cos, sin, sqrt, tan


def convert_agn_to_normal(chieff, chip, q, theta1, theta2, deltaphi):
    qfactor = q * ((4 * q) + 3) / (4 + (3 * q))
    kwargs = dict(
        chieff=chieff, chip=chip, q=q, Q=qfactor, Δϕ=deltaphi, θ1=theta1, θ2=theta2
    )

    case1 = (a1case1(**kwargs), a2case1(**kwargs))
    case2 = (a1case2(**kwargs), a2case2(**kwargs))
    if all(0 <= c <= 1 for c in case1):
        a1, a2 = case1[0], case1[1]
    else:
        a1, a2 = case2[0], case2[1]

    print(case1, case2)
    return a1, a2


def sec(x):
    return 1.0 / cos(x)


def a1case1(chieff, chip, q, Q, Δϕ, θ1, θ2):
    aux0 = chieff * (q * (
            (1. + q) * (Q * ((cos(Δϕ)) * ((cos(θ2)) * ((sin(θ2)) * (tan(θ1))))))))
    aux1 = (chip ** 2) * (q * (Q * ((cos(Δϕ)) * ((cos(θ2)) * ((sin(θ2)) * (tan(θ1)))))))
    aux2 = (chip ** 2) - ((chieff ** 2) * (
            (((1. + q) ** 2)) * ((((sin(Δϕ)) ** 2)) * (((tan(θ1)) ** 2)))))
    aux3 = ((chip ** 2) * ((q ** 2) * ((((cos(θ2)) ** 2)) * (((tan(θ1)) ** 2))))) + (
            (Q ** 2) * ((((sin(θ2)) ** 2)) * aux2))
    aux4 = (cos(θ2)) * ((((sec(θ1)) ** 2)) * (
        sqrt((((cos(θ1)) ** 4.) * ((-2. * aux1) + aux3)))))
    aux5 = (-(sec(θ1)) * ((aux0 + (q * aux4)) - (
            chieff * ((1. + q) * ((Q ** 2) * (((sin(θ2)) ** 2)))))))
    aux6 = (-2. * (q * (Q * ((cos(Δϕ)) * ((cos(θ2)) * ((sin(θ2)) * (tan(θ1)))))))) + (
            (q ** 2) * ((((cos(θ2)) ** 2)) * (((tan(θ1)) ** 2))))
    output = aux5 / (((Q ** 2) * (((sin(θ2)) ** 2))) + aux6)
    return output


def a1case2(chieff, chip, q, Q, Δϕ, θ1, θ2):
    aux0 = (chip ** 2) * (q * (Q * ((cos(Δϕ)) * ((cos(θ2)) * ((sin(θ2)) * (tan(θ1)))))))
    aux1 = (chip ** 2) - ((chieff ** 2) * (
            (((1. + q) ** 2)) * ((((sin(Δϕ)) ** 2)) * (((tan(θ1)) ** 2)))))
    aux2 = ((chip ** 2) * ((q ** 2) * ((((cos(θ2)) ** 2)) * (((tan(θ1)) ** 2))))) + (
            (Q ** 2) * ((((sin(θ2)) ** 2)) * aux1))
    aux3 = (cos(θ2)) * ((((sec(θ1)) ** 2)) * (
        sqrt((((cos(θ1)) ** 4.) * ((-2. * aux0) + aux2)))))
    aux4 = chieff * (q * (
            (1. + q) * (Q * ((cos(Δϕ)) * ((cos(θ2)) * ((sin(θ2)) * (tan(θ1))))))))
    aux5 = (sec(θ1)) * (((chieff * ((1. + q) * ((Q ** 2) * (((sin(θ2)) ** 2))))) + (
            q * aux3)) - aux4)
    aux6 = (-2. * (q * (Q * ((cos(Δϕ)) * ((cos(θ2)) * ((sin(θ2)) * (tan(θ1)))))))) + (
            (q ** 2) * ((((cos(θ2)) ** 2)) * (((tan(θ1)) ** 2))))
    output = aux5 / (((Q ** 2) * (((sin(θ2)) ** 2))) + aux6)
    return output


def a2case1(chieff, chip, q, Q, Δϕ, θ1, θ2):
    aux0 = Q * ((cos(Δϕ)) * ((cos(θ1)) * ((cos(θ2)) * ((sin(θ1)) * (sin(θ2))))))
    aux1 = ((chip ** 2) * (((cos(θ1)) ** 2))) - ((chieff ** 2) * (
            (((1. + q) ** 2)) * ((((sin(Δϕ)) ** 2)) * (((sin(θ1)) ** 2)))))
    aux2 = ((chip ** 2) * ((q ** 2) * ((((cos(θ2)) ** 2)) * (((sin(θ1)) ** 2))))) + (
            (-2. * ((chip ** 2) * (q * aux0))) + (
            (Q ** 2) * (aux1 * (((sin(θ2)) ** 2)))))
    aux3 = (chieff * (q * ((1. + q) * ((cos(θ2)) * (((sin(θ1)) ** 2)))))) + (
        sqrt(((((cos(θ1)) ** 2)) * aux2)))
    aux4 = chieff * (q * (Q * ((cos(Δϕ)) * ((cos(θ1)) * ((sin(θ1)) * (sin(θ2)))))))
    aux5 = (aux3 - aux4) - (
            chieff * (Q * ((cos(Δϕ)) * ((cos(θ1)) * ((sin(θ1)) * (sin(θ2)))))))
    aux6 = Q * ((cos(Δϕ)) * ((cos(θ1)) * ((cos(θ2)) * ((sin(θ1)) * (sin(θ2))))))
    aux7 = ((q ** 2) * ((((cos(θ2)) ** 2)) * (((sin(θ1)) ** 2)))) + (
            (-2. * (q * aux6)) + (
            (Q ** 2) * ((((cos(θ1)) ** 2)) * (((sin(θ2)) ** 2)))))
    output = aux5 / aux7
    return output


def a2case2(chieff, chip, q, Q, Δϕ, θ1, θ2):
    aux0 = Q * ((cos(Δϕ)) * ((cos(θ1)) * ((cos(θ2)) * ((sin(θ1)) * (sin(θ2))))))
    aux1 = ((chip ** 2) * (((cos(θ1)) ** 2))) - ((chieff ** 2) * (
            (((1. + q) ** 2)) * ((((sin(Δϕ)) ** 2)) * (((sin(θ1)) ** 2)))))
    aux2 = ((chip ** 2) * ((q ** 2) * ((((cos(θ2)) ** 2)) * (((sin(θ1)) ** 2))))) + (
            (-2. * ((chip ** 2) * (q * aux0))) + (
            (Q ** 2) * (aux1 * (((sin(θ2)) ** 2)))))
    aux3 = (chieff * (q * ((1. + q) * ((cos(θ2)) * (((sin(θ1)) ** 2)))))) - (
        sqrt(((((cos(θ1)) ** 2)) * aux2)))
    aux4 = chieff * (q * (Q * ((cos(Δϕ)) * ((cos(θ1)) * ((sin(θ1)) * (sin(θ2)))))))
    aux5 = (aux3 - aux4) - (
            chieff * (Q * ((cos(Δϕ)) * ((cos(θ1)) * ((sin(θ1)) * (sin(θ2)))))))
    aux6 = Q * ((cos(Δϕ)) * ((cos(θ1)) * ((cos(θ2)) * ((sin(θ1)) * (sin(θ2))))))
    aux7 = ((q ** 2) * ((((cos(θ2)) ** 2)) * (((sin(θ1)) ** 2)))) + (
            (-2. * (q * aux6)) + (
            (Q ** 2) * ((((cos(θ1)) ** 2)) * (((sin(θ2)) ** 2)))))
    output = aux5 / aux7
    return output
