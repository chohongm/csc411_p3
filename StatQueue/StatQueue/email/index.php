<?php
  $DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
  require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
  session_start();
  $username = $_POST['username'];
  $passwd = $_POST['passwd'];
  $action = $_REQUEST['action'];
  $account = $_REQUEST['account'];
  $messageid = $_GET['messageid'];
  
  $to = $_POST['to'];
  $cc = $_POST['cc'];
  $subject = $_POST['subject'];
  $message = $_POST['message'];
  
  $buttons = array();
  
?>