"""
% function y = x2y(x)
% function x = y2x(y)
%   x1 = theta1    y1 = chi_eff
%   x2 = theta2    y2 = chi_perpx
%   x3 = q         y3 = chi_perpy
%   x4 = chi1      y4 = chi1          <--- must be >0 to avoid inf error
%   x5 = chi2      y5 = chi2
%   x6 = phi1      y6 = theta2
%   x7 = phi2      y7 = q

% define trivial transformations, starting with q
y(7) = q;
% chi1
y(4) = chi1;
% chi2
y(5) = chi2;
% theta2
y(6) = theta2;

% effective parameters starting with chi_eff
tmp = chi1*cos(theta1) + q*chi2*cos(theta2);
y(1) = tmp / (1 + q);

% chi_perp_x
tmp =  chi1*sin(theta1)*cos(phi1) + q^2*chi2*sin(theta2)*cos(phi2);
y(2) = tmp / (1 + q^2*chi1*sin(theta1));

% chi_perp_y
tmp =  chi1*sin(theta1)*sin(phi1)+ q^2*chi2*sin(theta2)*sin(phi2);
y(3) = tmp / (1 + q^2*chi1*sin(theta1));

fprintf('add cases for chi1_perp < chi2_perp\n');

return

"""

from numpy import cos, sin, maximum, sqrt


def convert_normal_to_agn(theta1, theta2, q, chi1, chi2, deltaphi):
    c1z = zcomp(chi1, theta1)
    c2z = zcomp(chi2, theta2)

    chi_eff = (c1z + q * c2z) / (1 + q)

    qfactor = q * ((4 * q) + 3) / (4 + (3 * q))
    chi_p_term1 = chi1 * sin(theta1)
    chi_p_term2 = chi2 * sin(theta2) * qfactor
    chi_p_term3 = 2 * chi_p_term1 * chi_p_term2 * cos(deltaphi)
    chi_p = maximum(chi_p_term1, chi_p_term2)

    general_chi_p = sqrt(chi_p_term1**2 + chi_p_term2**2 + chi_p_term3)
    return chi_eff, general_chi_p


def xcomp(mag, theta, phi):
    return mag * sin(theta) * cos(phi)


def ycomp(mag, theta, phi):
    return mag * sin(theta) * sin(phi)


def zcomp(mag, theta):
    return mag * cos(theta)


def cartetian(mag, theta, phi):
    return (
        xcomp(mag, theta, phi),
        ycomp(mag, theta, phi),
        zcomp(mag, theta)
    )
