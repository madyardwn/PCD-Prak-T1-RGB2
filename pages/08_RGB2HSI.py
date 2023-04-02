# Author: - Achmadya Ridwan Ilyawan
#         - Fahmi Ahmad Fadilah
#         - Hilman Permana
# Date: 2023-04-02
# Description: Konversi warna citra dari RGB ke HSI

import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import math

# konversi warna RGB ke HSI
def RGB2HSI(I):
    Ir = I[:,:,0].astype(float)
    Ig = I[:,:,1].astype(float)
    Ib = I[:,:,2].astype(float)
    m, n = Ir.shape

    Ih = np.zeros((m, n), dtype=float)
    Is = np.zeros((m, n), dtype=float)
    Ii = np.zeros((m, n), dtype=float)

    for i in range(m):
        for j in range(n):
            r = Ir[i,j]/255.0
            g = Ig[i,j]/255.0
            b = Ib[i,j]/255.0

            minrgb = min(r, min(g, b))
            i2 = (r + g + b)/3.0
            if r == g and g == b:
                h = 0
                s = 0
            else:
                a = (r-g) + (r-b)
                rg2 = (r-g)*(r-g)
                rb2 = (r-b)*(g-b)
                b2 = math.sqrt(rg2+rb2)
                alpha = math.acos(0.5*(a/b2)) * 180/math.pi

                if g >= b:
                    h = alpha
                else:
                    h = 360 - alpha

                s = 1 - (3*(minrgb/(r+g+b)))

            Ih[i,j] = h/360.0
            Is[i,j] = s
            Ii[i,j] = i2

    Ihsi = np.zeros((m, n, 3), dtype=float)
    Ihsi[:,:,0] = Ih
    Ihsi[:,:,1] = Is
    Ihsi[:,:,2] = Ii

    return Ihsi



# judul halaman web
st.title('Konversi Warna Citra dari RGB ke HSI')

# upload gambar
uploaded_file = st.file_uploader("Upload Files", type=["jpg", "png", "jpeg"])

# menampilkan gambar
if uploaded_file is not None:

    image = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # menampilkan gambar
    colimg1, colimg2 = st.columns(2)

    with colimg1:
        st.image(image, caption='Gambar Asli', use_column_width=True)

    # konversi warna
    with colimg2:
        st.image(RGB2HSI(image), caption='Gambar Hasil Konversi', use_column_width=True)



# menampilkan kode sumber
tab1, tab2 = st.tabs(['MATLAB', 'Python'])

with tab1:
    st.markdown('''
    ```matlab    
    function f = RGB2HSI(I)
        Ir = double(I(:,:,1));
        Ig = double(I(:,:,2));
        Ib = double(I(:,:,3));
        [m,n] = size(Ir);

        for i = 1 : m
        for j = 1 : n
            r = Ir(i,j)/255;
            g = Ig(i,j)/255;
            b = Ib(i,j)/255;
            
            minrgb = min(r,min(g,b));
            i2 = (r+g+b)/3;
            if r==g && g==b 
                    h = 0;
                    s = 0;
            else
            
                a = (r-g)+(r-b);
                rg2 = (r-g)*(r-g);
                rb2 = (r-b)*(g-b);
                b2 = sqrt(double(rg2+rb2));
                alpha = acos(double(0.5*(a/b2))) * 180/pi;
                
                if g >= b
                    h = alpha;
                else
                    h = 360 - alpha;   
                end
                
                s = 1 - (3*(minrgb/(r+g+b)));
            end
            Ih(i,j) = double(h/360);
            Is(i,j) = double(s);
            Ii(i,j) = double(i2);
        end
        end
        Ihsi(:,:,1) = Ih;
        Ihsi(:,:,2) = Is;
        Ihsi(:,:,3) = Ii;
        
        % figure(1), imshow(Ihsi);
        f = Ihsi;
    end


    ```
    ''', unsafe_allow_html=True)


with tab2:
    st.markdown('''
    ```python
    # konversi warna RGB ke HSI
    def RGB2HSI(I):
        Ir = I[:,:,0].astype(float)
        Ig = I[:,:,1].astype(float)
        Ib = I[:,:,2].astype(float)
        m, n = Ir.shape

        Ih = np.zeros((m, n), dtype=float)
        Is = np.zeros((m, n), dtype=float)
        Ii = np.zeros((m, n), dtype=float)

        for i in range(m):
            for j in range(n):
                r = Ir[i,j]/255.0
                g = Ig[i,j]/255.0
                b = Ib[i,j]/255.0

                minrgb = min(r, min(g, b))
                i2 = (r + g + b)/3.0
                if r == g and g == b:
                    h = 0
                    s = 0
                else:
                    a = (r-g) + (r-b)
                    rg2 = (r-g)*(r-g)
                    rb2 = (r-b)*(g-b)
                    b2 = math.sqrt(rg2+rb2)
                    alpha = math.acos(0.5*(a/b2)) * 180/math.pi

                    if g >= b:
                        h = alpha
                    else:
                        h = 360 - alpha

                    s = 1 - (3*(minrgb/(r+g+b)))

                Ih[i,j] = h/360.0
                Is[i,j] = s
                Ii[i,j] = i2

        Ihsi = np.zeros((m, n, 3), dtype=float)
        Ihsi[:,:,0] = Ih
        Ihsi[:,:,1] = Is
        Ihsi[:,:,2] = Ii

        return Ihsi
    ```
    ''', unsafe_allow_html=True)