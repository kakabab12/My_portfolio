<% 
X = CInt(Request.Form("txtid"))
DSNless="DRIVER={Microsoft Access Driver (*.mdb)}; "
DSNless=DSNless & "DBQ=" & server.mappath("onchat.mdb")

Set Conn = Server.CreateObject("ADODB.Connection")
Conn.Open DSNless

Set Rs = Server.CreateObject("ADODB.Recordset")
Rs.Open "Select * From IdeaTime Where id = "& X &";", Conn

%>

<html>
<body>

  <form method="post" action="cupdate3.asp">
    <input type="text" name="txtIdea" value="<% = Rs("Idea")%>">
    <input type="hidden" name="txtid" value="<% = Rs("id")%>">
    <input type="submit" value="UPDATE">
  </form>

</body>
</html>

<% 
Rs.close
set Rs=nothing
Conn.close
Set Conn=nothing
%>