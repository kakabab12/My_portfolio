<%

Randomize

Com  = Int(Rnd()*3) + 1

Me1 = CInt(Request.Form("drpItem"))

If (Me1 = Com) Then
  Output = "Equal"
ElseIf ((Me1=1 AND Com=3) OR (Me1=2 AND Com=1) OR (Me1=3 AND Com=2)) Then
  Output = "Me Win"
Else
  Output = "Com Win"
End If

%>

<html>
<head>
  <meta charset="UTF-8">

</head>

<body>
  <% = Output %>
 
</body>
</html>



