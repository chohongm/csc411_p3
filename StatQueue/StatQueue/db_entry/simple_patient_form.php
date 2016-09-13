<?php
  $DOCUMENT_ROOT = $_SERVER['DOCUMENT_ROOT'];
  require_once("$DOCUMENT_ROOT/functions/statq_fns.php");
  do_header();

  display_simple_wl_form();
  
  do_footer();
?>