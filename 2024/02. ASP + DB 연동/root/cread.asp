<% 

DSNless="DRIVER={Microsoft Access Driver (*.mdb)}; "
DSNless=DSNless & "DBQ=" & server.mappath("onchat.mdb")

Set Conn = Server.CreateObject("ADODB.Connection")
Conn.Open DSNless

Set Rs = Server.CreateObject("ADODB.Recordset")
Rs.Open "Select * From IdeaTime Order By id DESC", Conn

%>

<html>
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="3">
</head>

<body>

<table border="0" width="800">

<%
while not Rs.eof
%>

  <tr <% If Rs("id") Mod 2 = 0 Then %> bgcolor="lightblue" <% Else %> bgcolor="beige" <% End If %> >
    <td> <% = Rs("Idea")%> </td>

    <td> 
      <form method="post" action="cdelete.asp">
        <input type="hidden" name="txtid" value="<% = Rs("id")%>">
        <input type="submit" value=" DELETE ">
      </form>
    </td>

    <td> 
      <form method="post" action="cupdate2.asp">
        <input type="hidden" name="txtid" value="<% = Rs("id")%>">
        <input type="submit" value=" UPDATE ">
      </form>
    </td>

  </tr>

<%
Rs.movenext
Wend
%>

</table>

</body>
</html>

<%

Rs.close
set Rs=nothing
Conn.close
Set Conn=nothing

%>


