<%

X = CInt(Request.Form("txtNum1"))
Y = CInt(Request.Form("txtNum2"))
Op = Request.Form("radOp")

'If Op = "+" Then
'  Z = X + Y
'ElseIf Op = "-" Then
'  Z = X - Y
'ElseIf Op = "*" Then
'  Z = X * Y
'Else
'  Z = X / Y
'End If

Select Case Op
	Case "+"
    Z = X + Y
	Case "-"
    Z = X - Y
	Case "*"
    Z = X * Y
	Case Else
    Z = X / Y
End Select


%>

<html>
<body>

 <font color="red" size="16"><% = Z %></font>

</body>
</html>