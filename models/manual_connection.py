from fyers_apiv3 import fyersModel
import webbrowser


### Input parameters
redirect_uri= "https://www.google.com"  ## redirect_uri you entered while creating APP.
client_id = "VWOF51Y3H0-100"            ## Client_id here refers to APP_ID of the created app
secret_key = "V46JR86CPS"               ## secret_key which you got after creating the app

grant_type = "authorization_code"
response_type = "code"
state = "sample"

### Connect to the sessionModel object here with the required input parameters
appSession = fyersModel.SessionModel(client_id = client_id, redirect_uri = redirect_uri, response_type=response_type, state=state, secret_key=secret_key, grant_type=grant_type)

### Make  a request to generate_authcode object this will return a login url which you need to open in your browser from where you can get the generated auth_code
generateTokenUrl = appSession.generate_authcode()

print((generateTokenUrl))
webbrowser.open(generateTokenUrl,new=1)

### After successful login the user can copy the generated auth_code over here and make the request to generate the accessToken
auth_code = input("Enter Auth Code: ")
appSession.set_token(auth_code)
response = appSession.generate_token()

### There can be two cases over here you can successfully get the acccessToken over the request or you might get some error over here. so to avoid that have this in try except block
try:
    access_token = response["access_token"]
    print("token: ",access_token)
except Exception as e:
    print(e,response)  ## This will help you in debugging then and there itself like what was the error and also you would be able to see the value you got in response variable. instead of getting key_error for unsuccessfull response.

fyers = fyersModel.FyersModel(token=access_token, is_async=False, client_id=client_id, log_path="../../")

#Get details about your account
response = fyers.get_profile()
print(response)

#save to txt file
with open("./client_id.txt", 'w') as file:
    file.write(client_id)

with open("./access_token.txt", 'w') as file:
    file.write(access_token)