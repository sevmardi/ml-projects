from django.contrib import admin
from django.urls import path, include
from django.contrib.auth.views import LoginView, PasswordResetView, LogoutView
from server.urls import router


urlpatterns = [
    path('', include('server.urls')),
    path('admin/', admin.site.urls),
    path('login/', LoginView.as_view(template_name='login.html',
                                     redirect_authenticated_user=True), name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('password_reset/', PasswordResetView.as_view(), name='password_reset'),
    path('api-auth/', include('rest_framework.urls')),
    path('api/', include(router.urls)),

]
