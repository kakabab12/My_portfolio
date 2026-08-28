<%

Randomize

Com  = Int(Rnd()*2) + 1

Me1 = CInt(Request.Form("txtNum1"))

If (Me1 = Com) Then
  Output = "Equal"
ElseIf  ((Me1=1) AND (Com = 2)) Then
  Output = "Com Win"
Else
  Output = "Me Win"
End If

'If ((Me1=2) AND (Com = 1)) Then
'  Output = "Me Win"
'End If

%>

<html>
<head>
  <meta charset="UTF-8">

</head>

<body>
  <% = Output %>
 
</body>
</html>



